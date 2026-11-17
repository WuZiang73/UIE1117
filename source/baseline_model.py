from dp_ema import SampleDropPath
from collections import namedtuple

import torch
import torch.nn as nn

from dark_variants import DarkPrior, build_dark_module
from dark_context_v2 import DarkContextV2
from dark_variants_v2 import build_dark_module_v2
from dino_c01 import DinoC01
from dino_dg71 import DinoOutputArbiter, DinoSemanticArbiter
from lowfreq import LowFreqResidual, LowFreqResidualGated
from psf_prefilter import WavelengthPreFilter, ISPProxy
from isp_out import ISPOutput
from psf_trainable import TrainablePSF
from psf_internal.module import InternalOpticalUpdate
from psf_physics import render_underwater
from psf_structural import PSFStructural
from psf_improvements import PSFImprovements
from dino_five_structural import (
    F1Bridge,
    F2Affinity,
    F3Slots,
    F4SkipRoute,
    F5DeformSkip,
)
from dino_wavelet_variants import DinoWaveletVariant
from dino_global_methods import (
    DinoEncoderCross,
    DinoGlobalAdaIN,
    DinoOutputMoE,
    DinoResidualBranch,
    DinoWaveletMod,
)
from dino_structure_methods import (
    DinoClusterGate,
    DinoDecoder2Cross,
    DinoDynamicConv,
    DinoPromptBottleneck,
    SemanticSkipFusion,
)
from dino_extra_methods import (
    DinoChannelGate,
    DinoColorGain,
    DinoDarkCross,
    DinoRoutingBias,
)
from global_semantic_modules import (
    ColorMatrixHead,
    GlobalSemanticPyramid,
    SemanticCrossAttention,
)

DG71_MODES = {
    "DG71G": "C01D1G",
    "DG71A": "C01D1A",
    "DG71R": "C01D1R",
}
DG70_MODES = ("DG70ARB", "DG70ARBP")


class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, in_channel, kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_channel, bias=bias,
        )
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class BasicBlock(nn.Module):
    """Plain DWS-HINB residual block (no wavelet / gradient innovations)."""
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.drop_path = SampleDropPath(0.0)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        return self.drop_path(out) + self.identity(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class BaselineUNet(nn.Module):
    """U-Net baseline with every paper innovation removed:
    no adaptive white balance, no WEB, no SGFB; Charbonnier loss only."""
    def __init__(self, in_channels=3, feature_channels=32, variant=None, num_clusters=3,
                 semantic=None, semantic_weights=None, semantic_layers=(2, 6, 10)):
        super().__init__()
        c = feature_channels
        self.variant = variant
        self.semantic_name = None if semantic in (None, "none", "L1") else semantic
        self.v2 = variant in ("M1v2", "M2v2", "M3v2", "M2v2R")
        if self.v2:
            self.dark_context_v2 = DarkContextV2()
            self.dark_dec1_v2 = build_dark_module_v2(variant, 2 * c)
        elif variant is not None:
            self.dark_prior = DarkPrior()
            self.dark_enc2 = build_dark_module(variant, 2 * c, num_clusters)
            self.dark_bot = build_dark_module(variant, 4 * c, num_clusters)
            self.dark_dec1 = build_dark_module(variant, 2 * c, num_clusters)
        self.first = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1)
        self.encoder1 = BasicBlock(c, c)
        self.down1 = Downsample(c)
        self.encoder2 = BasicBlock(2 * c, 2 * c)
        self.down2 = Downsample(2 * c)
        self.bottleneck = BasicBlock(4 * c, 4 * c)
        self.up1 = Upsample(4 * c)
        self.decoder1 = BasicBlock(2 * c, 2 * c)
        self.up2 = Upsample(2 * c)
        self.decoder2 = BasicBlock(c, c)
        self.out = nn.Conv2d(c, in_channels, kernel_size=3, stride=1, padding=1)
        # Construct semantic branches last so their frozen-teacher random init
        # cannot change the initialization of the C0 restoration backbone.
        self.semantic = None
        self.semantic_cross = None
        self.color_head = None
        self.dino_extra = None
        self.dino_c01 = None
        self.dino_dg71 = None
        self.dino_dg70 = None
        self.lowfreq = None
        self.prefilter = None
        self.isp_out = None
        self.psf_train = None
        self.internal = None
        self.structure = None
        self.psf = None
        self.psf_aux = None
        if self.semantic_name in DinoC01.MODES:
            self.dino_c01 = DinoC01(
                self.semantic_name,
                channels=2 * c,
                weights_path=semantic_weights,
                layers=semantic_layers,
            )
        elif self.semantic_name in DG71_MODES:
            self.dino_c01 = DinoC01(
                DG71_MODES[self.semantic_name],
                channels=2 * c,
                weights_path=semantic_weights,
                layers=semantic_layers,
            )
            self.dino_dg71 = DinoOutputArbiter("image")
            self.psf = PSFImprovements("DG7", e2_ch=2 * c, b0_ch=4 * c, out_ch=in_channels)
        elif self.semantic_name in DG70_MODES:
            self.dino_dg70 = DinoSemanticArbiter(
                "pixel" if self.semantic_name == "DG70ARBP" else "image",
                weights_path=semantic_weights,
                layers=semantic_layers,
            )
            self.psf = PSFImprovements("DG7", e2_ch=2 * c, b0_ch=4 * c, out_ch=in_channels)
        elif self.semantic_name in ("PF1", "PF2"):
            self.prefilter = WavelengthPreFilter() if self.semantic_name == "PF1" else ISPProxy()
        elif self.semantic_name in ("IOPT", "IOPT_D", "IOPT_ADAPT"):
            mode = {"IOPT": "full", "IOPT_D": "delta", "IOPT_ADAPT": "adapter"}[self.semantic_name]
            self.internal = InternalOpticalUpdate(feat_channels=2 * c, mode=mode)
        elif self.semantic_name in ("PSFM2", "PSFELL"):
            self.psf_train = TrainablePSF(elliptical=self.semantic_name == "PSFELL")
        elif self.semantic_name == "ISP3":
            self.isp_out = ISPOutput()
        elif self.semantic_name in ("LF32", "LF64", "LFG64"):
            if self.semantic_name == "LFG64":
                self.lowfreq = LowFreqResidualGated(size=64)
            elif self.semantic_name == "LF32":
                self.lowfreq = LowFreqResidual(size=32)
            else:
                self.lowfreq = LowFreqResidual(size=64)
        elif self.semantic_name in ("N0", "G1", "R1", "R2", "R3"):
            if self.semantic_name == "R1":
                ports = {"D1": 2 * c}
            elif self.semantic_name in ("R2", "R3"):
                ports = {"D1": 2 * c}
            else:
                ports = {"E2": 2 * c, "B0": 4 * c, "D1": 2 * c}
            self.semantic = GlobalSemanticPyramid(
                ports,
                mode=self.semantic_name,
                weights_path=semantic_weights,
                layers=semantic_layers,
            )
            self.semantic_cross = (
                SemanticCrossAttention(4 * c) if self.semantic_name == "R2" else None
            )
            self.color_head = (
                ColorMatrixHead(c, c) if self.semantic_name == "R3" else None
            )
        elif self.semantic_name in ("P1", "P2", "P3", "P4", "P5"):
            self.psf = PSFStructural(
                self.semantic_name, e2_ch=2 * c, b0_ch=4 * c,
                d1_ch=2 * c, out_ch=in_channels,
            )
        elif self.semantic_name in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q4F", "Q4C", "Q4B", "Q4S", "Q4M", "Q4R", "Q4Y", "Q4G", "Q4G2", "DG1", "DG2", "DG3", "DG4", "DG5", "DG6", "DG7", "DG7W32", "DG7W48", "DG7W52", "DG7W56", "DG7W60", "DG7W62", "DG7DW64", "DG7DW96", "DG7DW128", "DG7R128", "DG7R192", "DG7S64", "DG7S96", "DG8", "DG9", "V1", "V2", "W1", "W2", "W2N"):
            self.psf = PSFImprovements(
                self.semantic_name, e2_ch=2 * c, b0_ch=4 * c, out_ch=in_channels
            )
        elif self.semantic_name in ("F1", "F2", "F3", "F4", "F5"):
            if self.semantic_name == "F1":
                self.structure = F1Bridge(channels=4 * c, weights_path=semantic_weights)
            elif self.semantic_name == "F2":
                self.structure = F2Affinity(channels=4 * c, weights_path=semantic_weights)
            elif self.semantic_name == "F3":
                self.structure = F3Slots(channels=4 * c, weights_path=semantic_weights)
            elif self.semantic_name == "F4":
                self.structure = F4SkipRoute(channels=2 * c, weights_path=semantic_weights)
            else:
                self.structure = F5DeformSkip(channels=2 * c, weights_path=semantic_weights)
        elif self.semantic_name in ("U1", "U2", "U3", "U4", "U5"):
            self.structure = DinoWaveletVariant(
                channels=2 * c, variant=self.semantic_name, weights_path=semantic_weights
            )
        elif self.semantic_name in ("T1", "T2", "T3", "T4", "T5"):
            if self.semantic_name == "T1":
                self.structure = DinoResidualBranch(weights_path=semantic_weights)
            elif self.semantic_name == "T2":
                self.structure = DinoOutputMoE(feat_channels=c, weights_path=semantic_weights)
            elif self.semantic_name == "T3":
                self.structure = DinoEncoderCross(channels=2 * c, weights_path=semantic_weights)
            elif self.semantic_name == "T4":
                self.structure = DinoGlobalAdaIN(d1_channels=2 * c, d2_channels=c, weights_path=semantic_weights)
            else:
                self.structure = DinoWaveletMod(channels=2 * c, weights_path=semantic_weights)
        elif self.semantic_name in ("S1", "S2", "S3", "S4", "S5"):
            if self.semantic_name == "S1":
                self.structure = SemanticSkipFusion(
                    {"skip2": 2 * c, "skip1": c}, weights_path=semantic_weights
                )
            elif self.semantic_name == "S2":
                self.structure = DinoDynamicConv(channels=2 * c, weights_path=semantic_weights)
            elif self.semantic_name == "S3":
                self.structure = DinoPromptBottleneck(channels=4 * c, weights_path=semantic_weights)
            elif self.semantic_name == "S4":
                self.structure = DinoClusterGate(weights_path=semantic_weights)
            else:
                self.structure = DinoDecoder2Cross(channels=c, weights_path=semantic_weights)
        elif self.semantic_name in ("D1", "D2", "D3", "D4"):
            if self.semantic_name == "D1":
                self.dino_extra = DinoRoutingBias(weights_path=semantic_weights)
            elif self.semantic_name == "D2":
                self.dino_extra = DinoDarkCross(token_dim=6 + 16, weights_path=semantic_weights)
            elif self.semantic_name == "D3":
                self.dino_extra = DinoChannelGate(channels=2 * c, weights_path=semantic_weights)
            else:
                self.dino_extra = DinoColorGain(weights_path=semantic_weights)
        elif self.semantic_name is not None:
            raise ValueError(f"unsupported semantic variant {self.semantic_name}")

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if ".teacher." not in k}

    def load_state_dict(self, state_dict, strict=True):
        incoming = {k: v for k, v in state_dict.items() if ".teacher." not in k}
        own = self.state_dict()
        missing_all = [k for k in own if k not in incoming]
        unexpected_all = [k for k in incoming if k not in own]
        # PSF variants add/remove optional heads; keep old P/Q/V1 checkpoints loadable.
        missing = [k for k in missing_all if not k.startswith("psf.")]
        unexpected = [k for k in unexpected_all if not k.startswith("psf.")]
        if strict and (missing or unexpected):
            raise RuntimeError(f"state_dict mismatch missing={missing[:8]} unexpected={unexpected[:8]}")
        super().load_state_dict(incoming, strict=False)
        result = namedtuple("LoadResult", ["missing_keys", "unexpected_keys"])
        return result(missing_all, unexpected_all)

    def forward(self, x, return_pair=False):
        res = x
        if self.prefilter is not None:
            x = self.prefilter(x)
        dark_v2 = self.dark_context_v2(x) if self.v2 else None
        dark = self.dark_prior(x) if (self.variant is not None and not self.v2) else None
        sem = self.semantic(x) if self.semantic is not None else None
        if self.dino_c01 is not None:
            dino_aux = self.dino_c01(x)
        elif self.dino_dg70 is not None:
            dino_aux = self.dino_dg70(x)
        else:
            dino_aux = None
        extra = self.dino_extra(x) if self.dino_extra is not None else None
        struct = self.structure(x) if self.structure is not None else None
        use_ports = self.semantic_name in ("N0", "G1", "R1")
        x1 = self.encoder1(self.first(x))
        x2 = self.encoder2(self.down1(x1))
        if self.semantic_name == "T3":
            x2 = x2 + self.structure.apply(x2, struct)
        if sem is not None and use_ports and "E2" in self.semantic.channels:
            x2 = self.semantic.apply("E2", x2, sem["T"]["E2"])
        if self.variant is not None and not self.v2:
            x2 = self.dark_enc2(x2, dark)
        x3 = self.bottleneck(self.down2(x2))
        need_psf = self.psf is not None and not (self.semantic_name == "Q1" and not self.training)
        psf_aux = self.psf.calibrate(x, x2, x3) if need_psf else None
        if self.semantic_name in ("Q2", "Q4B"):
            x3 = self.psf.apply_b0(x3, psf_aux)
        if self.semantic_name == "P1":
            x3 = self.psf.apply_b0(x3, psf_aux)
        if sem is not None and use_ports and "B0" in self.semantic.channels:
            x3 = self.semantic.apply("B0", x3, sem["T"]["B0"])
        if sem is not None and self.semantic_name == "R2":
            x3 = self.semantic_cross(x3, sem["raw"])
        if self.semantic_name == "S3":
            x3 = self.structure.apply(x3, *struct)
        if self.semantic_name in ("F1", "F2", "F3"):
            x3 = self.structure.apply(x3, struct)
        if self.variant is not None and not self.v2:
            x3 = self.dark_bot(x3, dark)
        if self.semantic_name in ("P2", "P3", "P4", "Q3", "Q4", "Q5", "Q4F", "Q4C", "Q4S", "Q4M", "Q4G", "Q4G2", "DG1", "DG2"):
            q1 = self.up1(x3)
            skip2 = self.psf.apply_skip(x2, q1, psf_aux, x=x)
            y = self.decoder1(q1 + skip2)
        elif self.semantic_name in ("F4", "F5"):
            q1 = self.up1(x3)
            skip2 = self.structure.apply(x2, q1, struct)
            y = self.decoder1(q1 + skip2)
        elif self.semantic_name == "S1":
            q1 = self.up1(x3)
            skip2 = self.structure.apply("skip2", x2, q1, struct["T"]["skip2"])
            y = self.decoder1(q1 + skip2)
        else:
            q1 = self.up1(x3)
            skip = x2
            if self.internal is not None:
                delta, _ = self.internal(x, x2, q1)
                skip = skip + delta
            y = self.decoder1(q1 + skip)
        if self.semantic_name == "Q4Y":
            y = self.psf.film_y(y, x, psf_aux)
        if sem is not None and use_ports and "D1" in self.semantic.channels:
            y = self.semantic.apply("D1", y, sem["T"]["D1"])
        if self.semantic_name == "D3":
            y = y * extra
        if self.semantic_name == "S2":
            y = y + self.structure.apply(y, struct)
        if self.semantic_name == "T4":
            y = DinoGlobalAdaIN.apply(y, struct[0])
        if self.semantic_name == "T5" or self.semantic_name in ("U1", "U2", "U3", "U4", "U5"):
            y = self.structure.apply(y, struct)
        pre_dark = y
        if self.v2:
            if self.semantic_name == "D1":
                y = self.dark_dec1_v2(y, dark_v2, dino_logits=extra)
            elif self.semantic_name == "D2":
                y = self.dark_dec1_v2(
                    y, dark_v2, dino_tokens=extra, dino_cross=self.dino_extra.cross
                )
            elif self.semantic_name == "S4":
                y = self.dark_dec1_v2(y, dark_v2, cluster_gate=struct)
            else:
                y = self.dark_dec1_v2(y, dark_v2)
        elif self.variant is not None:
            y = self.dark_dec1(y, dark)
        if self.dino_c01 is not None:
            y = self.dino_c01.apply(pre_dark, y, dino_aux)
        if self.semantic_name == "S1":
            q2 = self.up2(y)
            skip1 = self.structure.apply("skip1", x1, q2, struct["T"]["skip1"])
            y = self.decoder2(q2 + skip1)
        else:
            y = self.up2(y) + x1
            y = self.decoder2(y)
        if self.semantic_name == "S5":
            y = y + self.structure.apply(y, struct)
        if self.semantic_name == "T4":
            y = DinoGlobalAdaIN.apply(y, struct[1])
        pred = self.out(y) + res
        if self.semantic_name == "P5":
            pred0 = pred
            pred, I0, I1, e0 = self.psf.postprocess(pred, y, x, psf_aux)
            psf_aux = dict(psf_aux)
            psf_aux.update({"I_hat": I0, "I_hat1": I1, "J0": pred0, "J1": pred, "e0": e0})
        elif self.semantic_name == "P1":
            psf_aux = dict(psf_aux)
            psf_aux["I_hat"] = render_underwater(pred, psf_aux)
        if self.semantic_name in ("Q1", "Q5") and self.training:
            psf_aux = dict(psf_aux)
            psf_aux["I_hat"] = self.psf.render(pred, psf_aux)
        if self.semantic_name == "V1":
            pred = self.psf.physics_residual(pred, x, psf_aux)
        if self.semantic_name == "V2":
            pred = self.psf.color_residual(pred, x, psf_aux)
        if self.semantic_name == "W1":
            pred = self.psf.residual_refine(pred, x, psf_aux)
        if self.semantic_name == "W2":
            pred = self.psf.affine_refine(pred, x, psf_aux)
        if self.semantic_name == "W2N":
            pred = self.psf.affine_refine_n(pred, x, psf_aux)
        if self.semantic_name in ("Q4R", "DG3", "DG4"):
            pred = self.psf.prior_residual(pred, x, psf_aux, self.semantic_name)
        if self.semantic_name == "DG5":
            pred = self.psf.dynamic_affine(pred, x, psf_aux)
        if self.semantic_name in ("DG6", "DG7", "DG7W32", "DG7W48", "DG7W52", "DG7W56", "DG7W60", "DG7W62", "DG7DW64", "DG7DW96", "DG7DW128", "DG7R128", "DG7R192", "DG7S64", "DG7S96") or self.semantic_name in DG71_MODES or self.semantic_name in DG70_MODES:
            pred_pre_dg7 = pred
            mode = "DG7" if (self.semantic_name in DG71_MODES or self.semantic_name in DG70_MODES) else self.semantic_name
            pred = self.psf.large_residual(pred, x, psf_aux, mode)
            if self.semantic_name in DG71_MODES:
                pred = self.dino_dg71(pred_pre_dg7, pred, dino_aux)
            elif self.semantic_name in DG70_MODES:
                pred = self.dino_dg70.apply(pred_pre_dg7, pred, dino_aux)
        if self.psf is not None:
            self.psf_aux = psf_aux
        if self.semantic_name == "T1":
            pred = pred + struct
        if self.semantic_name == "T2":
            pred = pred + self.structure.apply(y, struct)
        if sem is not None and self.semantic_name == "R3":
            pred = self.color_head(pred, y, sem["T"]["D1"])
        if self.semantic_name == "D4":
            pred = pred * (1.0 + extra[:, :3, None, None]) + extra[:, 3:, None, None]
        if self.lowfreq is not None:
            pred = self.lowfreq(pred, x)
        if self.isp_out is not None:
            pred = self.isp_out(pred)
        if self.psf_train is not None:
            pred = self.psf_train(pred)
        if return_pair:
            if self.semantic_name != "DG7":
                raise RuntimeError("return_pair is only implemented for DG7")
            return pred_pre_dg7, pred
        return pred
