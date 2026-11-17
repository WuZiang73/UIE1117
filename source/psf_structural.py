import torch
import torch.nn as nn
import torch.nn.functional as F

from psf_physics import PSFCalibrator, _apply_kernel, _gaussian_kernels, render_underwater


class PSFStructural(nn.Module):
    """P1-P5: structural uses of a low-dimensional wavelength/range PSF prior."""

    def __init__(self, mode, e2_ch=64, b0_ch=128, d1_ch=64, out_ch=3):
        super().__init__()
        assert mode in ("P1", "P2", "P3", "P4", "P5")
        self.mode = mode
        self.cal = PSFCalibrator(e2_ch=e2_ch, b0_ch=b0_ch)
        k = self.cal.kernel_size
        # P1
        self.p_l = nn.Conv2d(1, 16, 1)
        self.p_theta = nn.Linear(33, 16)
        self.p1_fuse = nn.Conv2d(b0_ch + 32, b0_ch, 1)
        self.p1_gate = nn.Parameter(torch.zeros(1))
        # P2
        basis = _gaussian_kernels(torch.tensor([[[1.0, 2.0, 3.0]]]), k)[0, 0]  # 3,k,k
        self.register_buffer("basis", basis[:, None], persistent=False)
        self.p2_adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(e2_ch, 16, 1), nn.GELU(), nn.Conv2d(16, e2_ch, 1))
            for _ in range(3)
        ])
        self.p2_coef = nn.Sequential(nn.Conv2d(4, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 3, 1))
        self.p2_gate = nn.Parameter(torch.zeros(1))
        # P3
        self.p3_local = nn.Sequential(
            nn.Conv2d(e2_ch, e2_ch, 3, 1, 1, groups=e2_ch), nn.GELU(), nn.Conv2d(e2_ch, e2_ch, 1)
        )
        self.p3_context = nn.Sequential(
            nn.Conv2d(e2_ch, e2_ch, 5, 1, 2, groups=e2_ch), nn.GELU(), nn.Conv2d(e2_ch, e2_ch, 1)
        )
        self.p3_route = nn.Sequential(
            nn.Conv2d(1 + 3 + 1 + e2_ch + e2_ch, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 3, 1)
        )
        nn.init.zeros_(self.p3_route[-1].weight); nn.init.zeros_(self.p3_route[-1].bias)
        self.p3_gate = nn.Parameter(torch.zeros(1))
        # P4
        self.p4_rgb = nn.Sequential(
            nn.Conv2d(6, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, e2_ch, 1)
        )
        self.p4_fuse = nn.Conv2d(e2_ch * 2, e2_ch, 1)
        self.p4_gate = nn.Parameter(torch.zeros(1))
        # P5
        self.p5_corr = nn.Sequential(
            nn.Conv2d(out_ch + out_ch + 1 + 3 + d1_ch // 2, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, out_ch, 3, 1, 1),
        )
        self.p5_gate = nn.Parameter(torch.zeros(1))

    def calibrate(self, x, e2, b0):
        return self.cal(x, e2, b0)

    def apply_b0(self, b0, aux):
        l64 = F.interpolate(aux["L"], size=b0.shape[-2:], mode="bilinear", align_corners=False)
        cond = torch.cat([
            self.p_l(l64), self.p_theta(aux["raw"])[:, :, None, None].expand(-1, -1, *b0.shape[-2:])
        ], dim=1)
        return b0 + self.p1_gate * self.p1_fuse(torch.cat([b0, cond], dim=1))

    def apply_skip(self, e2, q1, aux, x=None):
        if self.mode == "P2":
            coef = torch.sigmoid(self.p2_coef(torch.cat([
                aux["L"], aux["spread"][:, :, None, None].expand(-1, -1, 16, 16)
            ], dim=1)))
            coef = F.interpolate(coef, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            r = 0.0
            for i in range(3):
                b = F.conv2d(e2, self.basis[i].expand(e2.shape[1], 1, -1, -1),
                             padding=self.basis.shape[-1] // 2, groups=e2.shape[1])
                r = r + coef[:, i:i + 1] * self.p2_adapters[i](b)
            return e2 + self.p2_gate * (r - e2)
        if self.mode == "P3":
            logits = self.p3_route(torch.cat([
                aux["L"],
                aux["spread"][:, :, None, None].expand(-1, -1, 16, 16),
                aux["conf"][:, :, None, None].expand(-1, -1, 16, 16),
                F.avg_pool2d(e2, 8), F.avg_pool2d(q1, 8),
            ], dim=1))
            a = torch.softmax(logits, dim=1)
            a1 = F.interpolate(a[:, 1:2], size=e2.shape[-2:], mode="bilinear", align_corners=False) * self.p3_gate
            a2 = F.interpolate(a[:, 2:3], size=e2.shape[-2:], mode="bilinear", align_corners=False) * self.p3_gate
            return e2 + a1 * (self.p3_local(e2) - e2) + a2 * (self.p3_context(e2) - e2)
        if self.mode == "P4":
            rgb = x if x is not None else q1[:, :3]
            resp = _apply_kernel(rgb, aux["kernel_eff"])
            feat = self.p4_rgb(torch.cat([rgb, resp], dim=1))
            # q1 is 64ch at 128; PSF prior is fused into the skip value
            e2r = F.interpolate(feat, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.p4_fuse(torch.cat([e2, e2r], dim=1))
            return e2 + self.p4_gate * fused
        raise RuntimeError(self.mode)

    def postprocess(self, pred, y, x, aux):
        I0 = render_underwater(pred, aux)
        e0 = x - I0
        l = F.interpolate(aux["L"], size=pred.shape[-2:], mode="bilinear", align_corners=False)
        sp = F.interpolate(aux["spread"][:, :, None, None].expand(-1, -1, 16, 16),
                           size=pred.shape[-2:], mode="bilinear", align_corners=False)
        corr = self.p5_corr(torch.cat([pred, e0, l, sp, y], dim=1))
        j1 = pred + self.p5_gate * corr
        I1 = render_underwater(j1, aux)
        return j1, I0, I1, e0
