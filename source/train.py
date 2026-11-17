import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from baseline_model import BaselineUNet
from dino_extra_methods import dino_perceptual_loss
from semantic_backbone import FrozenDINOFeatures
from utils.dataset import get_loader
from utils.metrics import Evaluator


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))


def _worker_init(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UnderWaterDataset", args.dataset)
        self.evaluator = Evaluator()
        if args.arch == "paper":
            from model import myModel
            self.model = myModel(in_channels=3, feature_channels=32, use_white_balance=True)
        else:
            self.model = BaselineUNet(
                in_channels=3, feature_channels=32,
                variant=None if args.variant == "none" else args.variant,
                semantic=None if args.semantic == "none" else args.semantic,
                semantic_weights=args.dino_weights,
            )
        if args.init_backbone:
            sd = torch.load(args.init_backbone, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            allowed_missing = [k for k in missing if k.startswith(("dark_context_v2.", "dark_dec1_v2.", "semantic.", "semantic_cross.", "color_head.", "dino_extra.", "dino_c01.", "dino_dg71.", "dino_dg70.", "lowfreq.", "prefilter.", "isp_out.", "psf_train.", "internal.", "structure.", "psf."))]
            allowed_unexpected = [k for k in unexpected if k.startswith(("dark_context_v2.", "dark_dec1_v2.", "semantic.", "semantic_cross.", "color_head.", "dino_extra.", "dino_c01.", "dino_dg71.", "dino_dg70.", "lowfreq.", "prefilter.", "isp_out.", "psf_train.", "internal.", "structure.", "psf."))]
            if len(allowed_missing) != len(missing) or len(allowed_unexpected) != len(unexpected):
                raise RuntimeError(f"init key mismatch missing={missing} unexpected={unexpected}")
            print(f"[init] loaded {args.init_backbone} missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}", flush=True)
        if args.variant == "M2v2R" and hasattr(self.model, "dark_dec1_v2"):
            if not (0.0 <= args.rho <= 1.0):
                raise ValueError("rho must be in [0,1]")
            self.model.dark_dec1_v2.rho.fill_(args.rho)
        if args.freeze_backbone:
            for name, p in self.model.named_parameters():
                p.requires_grad = name.startswith(("psf.", "semantic.", "lowfreq.", "prefilter.", "isp_out.", "psf_train.", "internal."))
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in self.model.parameters())
            print(f"[freeze] trainable={n_train} / {n_all} (psf/semantic only)", flush=True)
        self.model = self.model.to(self.device)
        if args.loss == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif args.loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = CharbonnierLoss().to(self.device)
        self.psf_cycle_weight = float(args.psf_cycle)
        self.psf_diag = {}
        self.dino_loss_weight = float(args.dino_loss)
        self.dino_teacher = None
        if self.dino_loss_weight > 0:
            self.dino_teacher = FrozenDINOFeatures(weights_path=args.dino_weights).to(self.device)
            self.dino_teacher.eval()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.arch == "paper":
            self.out_dir = os.path.join(args.save_path, "paper_baseline", args.dataset, f"{stamp}_paper_seed{args.seed}")
        elif args.semantic != "none":
            tag = f"_{args.run_tag}" if args.run_tag else ""
            self.out_dir = os.path.join(
                args.save_path, f"{stamp}_{args.semantic}{tag}", f"{args.variant}_seed{args.seed}"
            )
        else:
            tag = f"_{args.run_tag}" if args.run_tag else ""
            self.out_dir = os.path.join(args.save_path, "baseline", args.dataset, f"{stamp}_{args.variant}{tag}_seed{args.seed}")
        os.makedirs(self.out_dir, exist_ok=True)
        meta = dict(vars(args))
        def _sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        meta["data_root_real"] = os.path.realpath(self.data_root)
        meta["init_sha256"] = _sha(args.init_backbone) if args.init_backbone else None
        meta["dino_weights"] = args.dino_weights if (args.semantic != "none" or args.dino_loss > 0) else None
        meta["dino_sha256"] = _sha(args.dino_weights) if meta["dino_weights"] and os.path.exists(args.dino_weights) else None
        meta["dino_loss_weight"] = args.dino_loss
        meta["psf_cycle_weight"] = args.psf_cycle
        npz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "dark_context_v2_uieb.npz")
        meta["npz_sha256"] = _sha(npz) if os.path.exists(npz) else None
        split_hash = {}
        for split in ("train", "val", "test"):
            d = os.path.join(self.data_root, split, "input")
            if os.path.isdir(d):
                h = hashlib.sha256()
                for name in sorted(os.listdir(d)):
                    h.update(name.encode())
                split_hash[split] = h.hexdigest()
        meta["split_name_sha256"] = split_hash
        meta["code_sha256"] = {f: _sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                               for f in ("train.py", "baseline_model.py", "dark_context_v2.py", "dark_variants_v2.py",
                                         "semantic_backbone.py", "global_semantic_modules.py", "dino_extra_methods.py", "dino_structure_methods.py", "dino_global_methods.py", "dino_wavelet_variants.py", "dino_five_structural.py")}
        vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "dinov2_vendor", "models", "vision_transformer.py")
        if os.path.exists(vendored):
            meta["dinov2_vendor_sha256"] = _sha(vendored)
        with open(os.path.join(self.out_dir, "run_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.optim = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optim, args.epoch, eta_min=args.lr * 1e-4
        )

        if args.resume:
            ckpt = torch.load(args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt)
            print("resumed", args.resume)

    def validation(self):
        self.evaluator.reset()
        loader = get_loader(
            os.path.join(self.data_root, "val"), self.args.eval_batch_size, 256,
            train=False, resize=True, num_workers=max(1, self.args.num_workers // 2),
            shuffle=False, pin_memory=True,
        )
        self.model.eval()
        with torch.no_grad():
            for i, (x, label, _) in tqdm(enumerate(loader), total=len(loader), leave=False):
                if self.args.max_val_batches and i >= self.args.max_val_batches:
                    break
                x = x.to(self.device)
                label = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                pred = torch.clamp(self.model(x), 0.0, 1.0)
                pred = pred.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                self.evaluator.evaluation(pred, label)
        ssim_, psnr_ = self.evaluator.getMean()
        self.model.train()
        return ssim_, psnr_

    def train(self):
        train_gen = torch.Generator()
        train_gen.manual_seed(self.args.seed + 1000)
        loader = get_loader(
            os.path.join(self.data_root, "train"), self.args.train_batch_size, 256,
            train=True, resize=True, num_workers=self.args.num_workers,
            shuffle=True, pin_memory=True,
            generator=train_gen, worker_init_fn=_worker_init,
        )
        best_psnr = 0.0
        self.model.train()
        for epoch in range(1, self.args.epoch + 1):
            if self.args.refiner_warmup > 0 and self.model.psf is not None:
                self.model.psf.refiner_scale = 1.0 if epoch > self.args.refiner_warmup else 0.0
            loss_sum, n = 0.0, 0
            loop = tqdm(enumerate(loader), total=len(loader), leave=False)
            for i, (x, label, _) in loop:
                if self.args.max_train_batches and i >= self.args.max_train_batches:
                    break
                x = x.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                if self.args.shot_noise > 0 or self.args.read_noise > 0:
                    var = self.args.shot_noise * x.clamp_min(0) + self.args.read_noise
                    x = (x + torch.randn_like(x) * var.sqrt()).clamp(0, 1)
                pred = self.model(x)
                loss = self.loss_fn(pred, label)
                if self.dino_teacher is not None:
                    loss = loss + self.dino_loss_weight * dino_perceptual_loss(
                        self.dino_teacher, pred, label
                    )
                aux = getattr(self.model, "psf_aux", None)
                if self.psf_cycle_weight > 0 and aux is not None and "I_hat" in aux:
                    warm = min(1.0, epoch / 10.0)
                    cyc = F.l1_loss(aux["I_hat"], x)
                    if "I_hat1" in aux:
                        cyc = cyc + F.l1_loss(aux["I_hat1"], x)
                    loss = loss + warm * self.psf_cycle_weight * cyc
                    self.psf_diag["cycle"] = self.psf_diag.get("cycle", 0.0) + float(cyc)
                if aux is not None:
                    for key in ("L", "beta", "kernel_sum", "spread"):
                        if key in aux:
                            self.psf_diag[key] = self.psf_diag.get(key, 0.0) + float(aux[key].mean())
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                n += 1
                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=loss.item(), lr=self.optim.param_groups[0]["lr"])
            avg_loss = loss_sum / max(n, 1)
            print(f"[{epoch}/{self.args.epoch}] loss={avg_loss:.6f} lr={self.optim.param_groups[0]['lr']:.3e}", flush=True)
            if self.psf_diag:
                denom = max(n, 1)
                diag = {k: v / denom for k, v in self.psf_diag.items()}
                diag["epoch"] = epoch
                with open(os.path.join(self.out_dir, "psf_diag.json"), "w") as f:
                    json.dump(diag, f, indent=2)
                self.psf_diag = {}
            if epoch % self.args.epoch_val == 0:
                ssim_, psnr_ = self.validation()
                print(f"[Validation] epoch={epoch} SSIM={ssim_:.4f} PSNR={psnr_:.4f}", flush=True)
                with open(os.path.join(self.out_dir, "records.txt"), "a") as f:
                    f.write(f"epoch={epoch} loss={avg_loss:.6f} PSNR={psnr_:.6f} SSIM={ssim_:.6f}\n")
                if psnr_ > best_psnr:
                    best_psnr = psnr_
                    torch.save(self.model.state_dict(), os.path.join(self.out_dir, "best_model.pth"))
                    with open(os.path.join(self.out_dir, "best.txt"), "w") as f:
                        f.write(f"epoch={epoch}\nPSNR={psnr_:.6f}\nSSIM={ssim_:.6f}\n")
                if self.args.smoke:
                    break
            self.scheduler.step()
            torch.save({
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "args": vars(self.args),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }, os.path.join(self.out_dir, "last_full.pth"))
        print("best PSNR:", best_psnr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--epoch_val", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train_batch_size", type=int, default=24)
    p.add_argument("--eval_batch_size", type=int, default=24)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    p.add_argument("--dataset", type=str, default="UIEB", choices=["UIEB", "LSUI", "EUVP-Scene", "EUVP-Dark"])
    p.add_argument("--variant", type=str, default="none", choices=["none", "M1", "M2", "M3", "M4", "M1v2", "M2v2", "M3v2", "M2v2R"])
    p.add_argument("--semantic", type=str, default="none", choices=["none", "N0", "G1", "R1", "R2", "R3", "L1", "D1", "D2", "D3", "D4", "S1", "S2", "S3", "S4", "S5", "T1", "T2", "T3", "T4", "T5", "U1", "U2", "U3", "U4", "U5", "F1", "F2", "F3", "F4", "F5", "P1", "P2", "P3", "P4", "P5", "Q1", "Q2", "Q3", "Q4", "Q5", "Q4F", "Q4C", "Q4B", "Q4S", "Q4M", "Q4R", "Q4Y", "Q4G", "Q4G2", "DG1", "DG2", "DG3", "DG4", "DG5", "DG6", "DG7", "DG7W32", "DG7W48", "DG7W52", "DG7W56", "DG7W60", "DG7W62", "DG7DW64", "DG7DW96", "DG7DW128", "DG7R128", "DG7R192", "DG7S64", "DG7S96", "DG8", "DG9", "C01D1G", "C01D1A", "C01D1R", "LF32", "LF64", "LFG64", "PF1", "PF2", "ISP3", "PSFM2", "PSFELL", "IOPT", "IOPT_D", "IOPT_ADAPT", "DG71G", "DG71A", "DG71R", "DG70ARB", "DG70ARBP", "V1", "V2", "W1", "W2", "W2N"])
    p.add_argument("--dino_loss", type=float, default=0.0)
    p.add_argument("--loss", type=str, default="charbonnier", choices=["charbonnier", "l1", "mse"])
    p.add_argument("--shot_noise", type=float, default=0.0)
    p.add_argument("--read_noise", type=float, default=0.0)
    p.add_argument("--refiner_warmup", type=int, default=0)
    p.add_argument("--psf_cycle", type=float, default=0.0)
    p.add_argument("--dino_weights", type=str, default="/data1/wza/third_party/dino_weights/vitb14.pth")
    p.add_argument("--arch", type=str, default="stripped", choices=["stripped", "paper"])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_backbone", type=str, default=None)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--freeze_backbone", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epoch = 1
        args.epoch_val = 1
        args.train_batch_size = 2
        args.eval_batch_size = 2
        args.num_workers = 0
        args.max_train_batches = 2
        args.max_val_batches = 2
    seed_everything(args.seed)
    print(json.dumps(vars(args), indent=2))
    t0 = time.time()
    Trainer(args).train()
    print("elapsed", time.time() - t0, "s", flush=True)


if __name__ == "__main__":
    main()
