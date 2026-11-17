import torch
import torch.nn as nn
import torch.nn.functional as F

from psf_physics import PSFCalibrator, _apply_kernel, render_underwater


class PSFImprovements(nn.Module):
    """Q1-Q5 improvements around P1 (cycle/bottleneck) and P4 (wavelength PSF).

    Q1: cycle-only calibrator (no inference change)
    Q2: L-only multiplicative B0 gate
    Q3: range-bin weighted RGB PSF response at E2 skip
    Q4: wavelength transmission/backscatter prior at E2 skip
    Q5: Q1 cycle + Q3 range-RGB skip response
    V1: single-point inverse underwater formation residual (PSF prior, no cycle)
    V2: single-point PSF-conditioned global color/tone residual (frozen C0)
    W1: single-point full-resolution PSF-conditioned residual refiner
    W2: single-point full-resolution PSF-conditioned local affine refiner
    W2N: W2 control without PSF maps (ablation only)
    Q4F: Q4 transmission/backscatter prior as E2 FiLM modulation
    Q4C: Q4 transmission/backscatter correction residual at E2
    Q4B: Q4 transmission/backscatter prior correction at B0
    Q4S: Q4 transmission/backscatter spatial gate on E2
    Q4M: Q4 multiplicative prior modulation on E2
    Q4R: Q4 transmission/backscatter prior output residual
    Q4Y: Q4 transmission/backscatter prior FiLM at decoder1
    Q4G: Q4F with fixed low-range reliability gate
    Q4G2: Q4F with low-range and low-backscatter reliability gate
    DG1: single-network E2 spatial gate, range-gated
    DG2: single-network E2 FiLM, range+backscatter-gated
    DG3: single-network output residual, range-gated
    DG4: single-network output residual, range+backscatter-gated
    DG5: single-network output affine, range+backscatter+learned gate
    DG6: single-network larger residual refiner, range+backscatter gate
    DG7: single-network larger residual refiner, range gate
    DG7W32: DG7 refiner with 32 channels
    DG7W48: DG7 refiner with 48 channels
    DG7W52: DG7 refiner with 52 channels
    DG7W56: DG7 refiner with 56 channels
    DG7W60: DG7 refiner with 60 channels
    DG7W62: DG7 refiner with 62 channels
    DG7DW64: DG7 refiner with depthwise-separable 64 channels
    DG7DW96: DG7 refiner with depthwise-separable 96 channels
    DG7R128: DG7 refiner at 128 resolution, 64 channels
    DG7R192: DG7 refiner at 192 resolution, 64 channels
    DG7S64: single 3x3 64-channel DG7 refiner
    DG7S96: single 3x3 96-channel DG7 refiner
    DG7DW128: DG7 refiner with depthwise-separable 128 channels
    DG8: DG7 + gentle high-backscatter gate (0.30)
    DG9: DG7 + gentler high-backscatter gate (0.40)
    """

    def __init__(self, mode, e2_ch=64, b0_ch=128, out_ch=3):
        super().__init__()
        assert mode in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q4F", "Q4C", "Q4B", "Q4S", "Q4M", "Q4R", "Q4Y", "Q4G", "Q4G2", "DG1", "DG2", "DG3", "DG4", "DG5", "DG6", "DG7", "DG7W32", "DG7W48", "DG7W52", "DG7W56", "DG7W60", "DG7W62", "DG7DW64", "DG7DW96", "DG7DW128", "DG7R128", "DG7R192", "DG7S64", "DG7S96", "DG8", "DG9", "V1", "V2", "W1", "W2", "W2N")
        self.mode = mode
        self.gate_th = 0.48
        self.refiner_scale = 1.0
        self.cal = PSFCalibrator(e2_ch=e2_ch, b0_ch=b0_ch)
        # Q2
        self.q2_conv = nn.Conv2d(1, b0_ch, 1)
        self.q2_gate = nn.Parameter(torch.zeros(1))
        # Q3/Q5
        self.q3_rgb = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, e2_ch, 1)
        )
        self.q3_fuse = nn.Conv2d(e2_ch * 2, e2_ch, 1)
        self.q3_gate = nn.Parameter(torch.zeros(1))
        # Q4
        self.q4_rgb = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, e2_ch, 1)
        )
        self.q4_fuse = nn.Conv2d(e2_ch * 2, e2_ch, 1)
        self.q4_gate = nn.Parameter(torch.zeros(1))
        # Q4F: local prior -> FiLM on E2
        self.q4f_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 3 + 1, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, e2_ch * 2, 1),
        )
        nn.init.zeros_(self.q4f_head[-1].weight)
        nn.init.zeros_(self.q4f_head[-1].bias)
        # Q4C: (1-T)(x-B) correction residual
        self.q4c_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, e2_ch, 1),
        )
        nn.init.zeros_(self.q4c_head[-1].weight)
        nn.init.zeros_(self.q4c_head[-1].bias)
        self.q4b_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 1, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, b0_ch, 1),
        )
        nn.init.zeros_(self.q4b_head[-1].weight)
        nn.init.zeros_(self.q4b_head[-1].bias)
        self.q4s_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 1, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, e2_ch, 1),
        )
        nn.init.zeros_(self.q4s_head[-1].weight)
        nn.init.zeros_(self.q4s_head[-1].bias)
        self.q4m_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, e2_ch, 1),
        )
        nn.init.zeros_(self.q4m_head[-1].weight)
        nn.init.zeros_(self.q4m_head[-1].bias)
        self.q4r_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 3 + 1, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, out_ch, 3, 1, 1),
        )
        nn.init.zeros_(self.q4r_head[-1].weight)
        nn.init.zeros_(self.q4r_head[-1].bias)
        self.q4y_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 3 + 1, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, e2_ch * 2, 1),
        )
        nn.init.zeros_(self.q4y_head[-1].weight)
        nn.init.zeros_(self.q4y_head[-1].bias)
        # V1: one scalar gate on the physically inverted input residual
        self.v1_gate = nn.Parameter(torch.zeros(1))
        # V2: one global affine residual conditioned on PSF parameters
        self.v2_head = nn.Sequential(
            nn.Linear(33 + 6 + 6 + 9, 64), nn.GELU(), nn.Linear(64, 6)
        )
        nn.init.zeros_(self.v2_head[-1].weight)
        nn.init.zeros_(self.v2_head[-1].bias)
        # W1: local residual refiner conditioned on physical maps
        self.w1_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 3 + 1 + 3 + 3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, out_ch, 3, 1, 1),
        )
        nn.init.zeros_(self.w1_head[-1].weight)
        nn.init.zeros_(self.w1_head[-1].bias)
        # W2: local affine maps conditioned on physical maps
        self.w2_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 1 + 3 + 3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 6, 3, 1, 1),
        )
        nn.init.zeros_(self.w2_head[-1].weight)
        nn.init.zeros_(self.w2_head[-1].bias)
        self.w2n_head = nn.Sequential(
            nn.Conv2d(3 + 3, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 6, 3, 1, 1),
        )
        nn.init.zeros_(self.w2n_head[-1].weight)
        nn.init.zeros_(self.w2n_head[-1].bias)
        self.dg5_gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.dg5_gate[-1].weight)
        nn.init.zeros_(self.dg5_gate[-1].bias)
        self.dg67_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 1 + 3 + 3 + 3 + 3, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg67_head[-1].weight)
        nn.init.zeros_(self.dg67_head[-1].bias)
        self.dg7w32_head = nn.Sequential(
            nn.Conv2d(19, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w32_head[-1].weight)
        nn.init.zeros_(self.dg7w32_head[-1].bias)
        self.dg7w48_head = nn.Sequential(
            nn.Conv2d(19, 48, 3, 1, 1), nn.GELU(),
            nn.Conv2d(48, 48, 3, 1, 1), nn.GELU(),
            nn.Conv2d(48, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w48_head[-1].weight)
        nn.init.zeros_(self.dg7w48_head[-1].bias)
        self.dg7w52_head = nn.Sequential(
            nn.Conv2d(19, 52, 3, 1, 1), nn.GELU(),
            nn.Conv2d(52, 52, 3, 1, 1), nn.GELU(),
            nn.Conv2d(52, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w52_head[-1].weight)
        nn.init.zeros_(self.dg7w52_head[-1].bias)
        self.dg7w56_head = nn.Sequential(
            nn.Conv2d(19, 56, 3, 1, 1), nn.GELU(),
            nn.Conv2d(56, 56, 3, 1, 1), nn.GELU(),
            nn.Conv2d(56, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w56_head[-1].weight)
        nn.init.zeros_(self.dg7w56_head[-1].bias)
        self.dg7w60_head = nn.Sequential(
            nn.Conv2d(19, 60, 3, 1, 1), nn.GELU(),
            nn.Conv2d(60, 60, 3, 1, 1), nn.GELU(),
            nn.Conv2d(60, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w60_head[-1].weight)
        nn.init.zeros_(self.dg7w60_head[-1].bias)
        self.dg7w62_head = nn.Sequential(
            nn.Conv2d(19, 62, 3, 1, 1), nn.GELU(),
            nn.Conv2d(62, 62, 3, 1, 1), nn.GELU(),
            nn.Conv2d(62, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7w62_head[-1].weight)
        nn.init.zeros_(self.dg7w62_head[-1].bias)
        self.dg7dw64_head = nn.Sequential(
            nn.Conv2d(19, 64, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=64), nn.GELU(),
            nn.Conv2d(64, 64, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=64), nn.GELU(),
            nn.Conv2d(64, 3, 1),
        )
        nn.init.zeros_(self.dg7dw64_head[-1].weight)
        nn.init.zeros_(self.dg7dw64_head[-1].bias)
        self.dg7dw96_head = nn.Sequential(
            nn.Conv2d(19, 96, 1), nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 1, groups=96), nn.GELU(),
            nn.Conv2d(96, 96, 1), nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 1, groups=96), nn.GELU(),
            nn.Conv2d(96, 3, 1),
        )
        nn.init.zeros_(self.dg7dw96_head[-1].weight)
        nn.init.zeros_(self.dg7dw96_head[-1].bias)
        self.dg7dw128_head = nn.Sequential(
            nn.Conv2d(19, 128, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128), nn.GELU(),
            nn.Conv2d(128, 3, 1),
        )
        nn.init.zeros_(self.dg7dw128_head[-1].weight)
        nn.init.zeros_(self.dg7dw128_head[-1].bias)
        self.dg7r128_head = nn.Sequential(
            nn.Conv2d(19, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7r128_head[-1].weight)
        nn.init.zeros_(self.dg7r128_head[-1].bias)
        self.dg7s64_head = nn.Sequential(
            nn.Conv2d(19, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7s64_head[-1].weight)
        nn.init.zeros_(self.dg7s64_head[-1].bias)
        self.dg7s96_head = nn.Sequential(
            nn.Conv2d(19, 96, 3, 1, 1), nn.GELU(),
            nn.Conv2d(96, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7s96_head[-1].weight)
        nn.init.zeros_(self.dg7s96_head[-1].bias)
        self.dg7r192_head = nn.Sequential(
            nn.Conv2d(19, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.dg7r192_head[-1].weight)
        nn.init.zeros_(self.dg7r192_head[-1].bias)

    def calibrate(self, x, e2, b0):
        return self.cal(x, e2, b0)

    def apply_b0(self, b0, aux):
        l64 = F.interpolate(aux["L"], size=b0.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "Q4B":
            size = b0.shape[-2:]
            L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
            T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
            B = aux["B"][:, :, None, None].expand(-1, -1, *size)
            feat = self.q4b_head(torch.cat([T, B, L], dim=1))
            return b0 + 0.25 * torch.tanh(feat)
        scale = 1.0 + self.q2_gate * 0.1 * torch.tanh(self.q2_conv(l64))
        return b0 * scale

    def apply_skip(self, e2, q1, aux, x):
        if self.mode in ("Q3", "Q5"):
            r = 0.0
            for bi in range(4):
                mb = F.interpolate(aux["m"][:, bi:bi + 1], size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
                r = r + mb * _apply_kernel(x, aux["kernel"][:, :, bi])
            feat = self.q3_rgb(r)
            return e2 + self.q3_gate * self.q3_fuse(torch.cat([e2, feat], dim=1))
        if self.mode == "Q4":
            L = F.interpolate(aux["L"], size=x.shape[-2:], mode="bilinear", align_corners=False)
            T = torch.exp(-aux["beta"][:, :, None, None] * L)
            prior = x * T + aux["B"][:, :, None, None] * (1.0 - T)
            feat = self.q4_rgb(prior)
            return e2 + self.q4_gate * self.q4_fuse(torch.cat([e2, feat], dim=1))
        if self.mode in ("Q4F", "Q4C", "Q4S", "Q4M", "Q4G", "Q4G2", "DG1", "DG2"):
            size = e2.shape[-2:]
            L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
            T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
            B = aux["B"][:, :, None, None].expand(-1, -1, *size)
            xr = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            prior = xr * T + B * (1.0 - T)
            mr = torch.sigmoid((0.48 - L) / 0.03)
            mb = torch.sigmoid((0.20 - B.mean(dim=1, keepdim=True)) / 0.10)
            if self.mode == "DG1":
                return e2 * (1.0 + 0.5 * mr * torch.tanh(self.q4s_head(torch.cat([T, B, L], dim=1))))
            if self.mode == "DG2":
                gb = self.q4f_head(torch.cat([prior, T, B, L], dim=1))
                gamma, beta = gb.chunk(2, dim=1)
                m = mr * mb
                return e2 * (1.0 + 0.5 * m * torch.tanh(gamma)) + 0.1 * m * torch.tanh(beta)
            if self.mode in ("Q4G", "Q4G2"):
                gb = self.q4f_head(torch.cat([prior, T, B, L], dim=1))
                gamma, beta = gb.chunk(2, dim=1)
                m = torch.sigmoid((0.46 - L) / 0.02)
                if self.mode == "Q4G2":
                    m = m * torch.exp(-2.0 * B.mean(dim=1, keepdim=True))
                return e2 * (1.0 + 0.5 * m * torch.tanh(gamma)) + 0.1 * m * torch.tanh(beta)
            if self.mode == "Q4F":
                gb = self.q4f_head(torch.cat([prior, T, B, L], dim=1))
                gamma, beta = gb.chunk(2, dim=1)
                return e2 * (1.0 + 0.5 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
            if self.mode == "Q4C":
                corr = (1.0 - T) * (xr - B)
                return e2 + 0.25 * torch.tanh(self.q4c_head(corr))
            if self.mode == "Q4S":
                return e2 * (1.0 + 0.5 * torch.tanh(self.q4s_head(torch.cat([T, B, L], dim=1))))
            return e2 * (1.0 + 0.5 * torch.tanh(self.q4m_head(prior)))
        raise RuntimeError(self.mode)

    def physics_residual(self, pred, x, aux):
        """Single-point prior: invert I = T*(K*J) + B*(1-T) on the input.

        K*J is approximated by the input I for the inverse step, so only
        T=exp(-beta*L) and B are used.  With gate=0 the output is exactly C0.
        """
        L = F.interpolate(aux["L"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None]
        j0 = ((x - B * (1.0 - T)) / T).clamp(0.0, 1.0)
        return pred + self.v1_gate * (j0 - x)

    def color_residual(self, pred, x, aux):
        """PSF-conditioned per-image affine residual, identity at init."""
        lmean = aux["L"].mean(dim=(2, 3))
        tglob = torch.exp(-aux["beta"] * lmean).clamp(0.05, 1.0)
        feat = torch.cat([
            aux["raw"],
            x.mean(dim=(2, 3)), x.std(dim=(2, 3), unbiased=False),
            pred.mean(dim=(2, 3)), pred.std(dim=(2, 3), unbiased=False),
            aux["beta"], aux["B"], tglob,
        ], dim=1)
        delta = self.v2_head(feat)
        scale = 1.0 + 0.25 * torch.tanh(delta[:, :3])
        bias = 0.10 * torch.tanh(delta[:, 3:])
        return pred * scale[:, :, None, None] + bias[:, :, None, None]

    def residual_refine(self, pred, x, aux):
        L = F.interpolate(aux["L"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *x.shape[-2:])
        feat = torch.cat([pred, x, pred - x, L, T, B], dim=1)
        delta = 0.25 * torch.tanh(self.w1_head(feat))
        return pred + delta

    def affine_refine(self, pred, x, aux):
        L = F.interpolate(aux["L"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *x.shape[-2:])
        delta = self.w2_head(torch.cat([pred, x, L, T, B], dim=1))
        scale = 1.0 + 0.25 * torch.tanh(delta[:, :3])
        bias = 0.10 * torch.tanh(delta[:, 3:])
        return pred * scale + bias

    def affine_refine_n(self, pred, x, aux):
        delta = self.w2n_head(torch.cat([pred, x], dim=1))
        scale = 1.0 + 0.25 * torch.tanh(delta[:, :3])
        bias = 0.10 * torch.tanh(delta[:, 3:])
        return pred * scale + bias

    def film_y(self, y, x, aux):
        size = y.shape[-2:]
        L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *size)
        xr = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        prior = xr * T + B * (1.0 - T)
        gb = self.q4y_head(torch.cat([prior, T, B, L], dim=1))
        gamma, beta = gb.chunk(2, dim=1)
        return y * (1.0 + 0.5 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

    def prior_residual(self, pred, x, aux, mode="Q4R"):
        size = x.shape[-2:]
        L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *size)
        prior = x * T + B * (1.0 - T)
        mr = torch.sigmoid((0.48 - L) / 0.03)
        mb = torch.sigmoid((0.20 - B.mean(dim=1, keepdim=True)) / 0.10)
        if mode == "DG3":
            return pred + 0.25 * mr * torch.tanh(self.q4r_head(torch.cat([pred, x, prior, L], dim=1)))
        if mode == "DG4":
            return pred + 0.25 * mr * mb * torch.tanh(self.q4r_head(torch.cat([pred, x, prior, L], dim=1)))
        return pred + 0.25 * torch.tanh(self.q4r_head(torch.cat([pred, x, prior, L], dim=1)))

    def large_residual(self, pred, x, aux, mode):
        size = x.shape[-2:]
        L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *size)
        prior = x * T + B * (1.0 - T)
        corr = (1.0 - T) * (x - B)
        mr = torch.sigmoid((self.gate_th - L) / 0.03)
        mb = torch.sigmoid((0.20 - B.mean(dim=1, keepdim=True)) / 0.10)
        if mode == "DG6":
            m = mr * mb
        elif mode == "DG8":
            m = mr * torch.sigmoid((0.30 - B.mean(dim=1, keepdim=True)) / 0.10)
        elif mode == "DG9":
            m = mr * torch.sigmoid((0.40 - B.mean(dim=1, keepdim=True)) / 0.15)
        else:
            m = mr
        if mode in ("DG7R128", "DG7R192"):
            size2 = (x.shape[-2] // 2, x.shape[-1] // 2) if mode == "DG7R128" else (192, 192)
            pred2 = F.interpolate(pred, size=size2, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x, size=size2, mode="bilinear", align_corners=False)
            L2 = F.interpolate(aux["L"], size=size2, mode="bilinear", align_corners=False)
            T2 = torch.exp(-aux["beta"][:, :, None, None] * L2).clamp(0.05, 1.0)
            B2 = aux["B"][:, :, None, None].expand(-1, -1, *size2)
            prior2 = x2 * T2 + B2 * (1.0 - T2)
            corr2 = (1.0 - T2) * (x2 - B2)
            head = self.dg7r128_head if mode == "DG7R128" else self.dg7r192_head
            delta = head(torch.cat([pred2, x2, L2, T2, B2, prior2, corr2], dim=1))
            delta = F.interpolate(delta, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return pred + 0.25 * self.refiner_scale * m * torch.tanh(delta)
        if mode == "DG7W32":
            head = self.dg7w32_head
        elif mode == "DG7W48":
            head = self.dg7w48_head
        elif mode == "DG7W52":
            head = self.dg7w52_head
        elif mode == "DG7W56":
            head = self.dg7w56_head
        elif mode == "DG7W60":
            head = self.dg7w60_head
        elif mode == "DG7W62":
            head = self.dg7w62_head
        elif mode == "DG7DW64":
            head = self.dg7dw64_head
        elif mode == "DG7DW96":
            head = self.dg7dw96_head
        elif mode == "DG7DW128":
            head = self.dg7dw128_head
        elif mode == "DG7S64":
            head = self.dg7s64_head
        elif mode == "DG7S96":
            head = self.dg7s96_head
        else:
            head = self.dg67_head
        delta = head(torch.cat([pred, x, L, T, B, prior, corr], dim=1))
        return pred + 0.25 * self.refiner_scale * m * torch.tanh(delta)

    def dynamic_affine(self, pred, x, aux):
        size = x.shape[-2:]
        L = F.interpolate(aux["L"], size=size, mode="bilinear", align_corners=False)
        T = torch.exp(-aux["beta"][:, :, None, None] * L).clamp(0.05, 1.0)
        B = aux["B"][:, :, None, None].expand(-1, -1, *size)
        mr = torch.sigmoid((0.48 - L) / 0.03)
        mb = torch.sigmoid((0.20 - B.mean(dim=1, keepdim=True)) / 0.10)
        feat = torch.cat([
            aux["L"].mean(dim=(2, 3)), aux["B"].mean(dim=1, keepdim=True), aux["conf"],
        ], dim=1)
        mg = 2.0 * torch.sigmoid(self.dg5_gate(feat))[:, :, None, None]
        m = mr * mb * mg
        delta = self.w2_head(torch.cat([pred, x, L, T, B], dim=1))
        scale = 1.0 + 0.25 * torch.tanh(delta[:, :3])
        bias = 0.10 * torch.tanh(delta[:, 3:])
        return pred * (1.0 + (scale - 1.0) * m) + bias * m

    def render(self, pred, aux):
        return render_underwater(pred, aux)
