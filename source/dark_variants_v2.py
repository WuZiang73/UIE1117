import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

def _match(ctx, size):
    out = {}
    for k in ("z", "w", "mask", "r"):
        v = ctx[k]
        if v.shape[-2:] != size:
            v = F.interpolate(v, size=size, mode="bilinear", align_corners=False)
        out[k] = v
    return out

def _cluster_tokens(z, w, f):
    # z:[B,6,H,W], w:[B,K,H,W], f:[B,C,H,W]
    mass = w.sum(dim=(2, 3), keepdim=True) + EPS           # B,K,1,1
    z_mean = (w.unsqueeze(2) * z.unsqueeze(1)).sum(dim=(3, 4), keepdim=True) / mass.unsqueeze(2)
    f_mean = (w.unsqueeze(2) * f.unsqueeze(1)).sum(dim=(3, 4), keepdim=True) / mass.unsqueeze(2)
    return torch.cat([z_mean.squeeze(-1).squeeze(-1), f_mean.squeeze(-1).squeeze(-1)], dim=-1)

def _zero_last(seq):
    last = seq[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)

class M1V2(nn.Module):
    def __init__(self, channels=64, hidden=16):
        super().__init__()
        self.gn = nn.GroupNorm(1, channels, affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(6 + channels, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        _zero_last(self.mlp)

    def forward(self, f, ctx):
        ctx = _match(ctx, f.shape[-2:])
        t = _cluster_tokens(ctx["z"], ctx["w"], f)       # B,K,6+C
        u = self.mlp(t).view(f.shape[0], ctx["w"].shape[1], 2, f.shape[1])
        gamma = 0.1 * torch.tanh(u[:, :, 0])
        beta = 0.1 * torch.tanh(u[:, :, 1])
        base = self.gn(f).unsqueeze(1)
        delta = (ctx["w"].unsqueeze(2) * (gamma.unsqueeze(-1).unsqueeze(-1) * base
                                          + beta.unsqueeze(-1).unsqueeze(-1))).sum(1)
        return f + delta

class _ExpertColor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, u):
        return self.net(u)

class _ExpertContext(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 0, groups=c)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, u):
        u = F.pad(u, (2, 2, 2, 2), mode="reflect")
        return self.pw(self.act(self.dw(u)))

class _ExpertDetail(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 0, groups=c)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, u):
        up = F.pad(u, (1, 1, 1, 1), mode="reflect")
        lo = F.avg_pool2d(up, 3, 1, 0)
        h = u - lo
        return self.pw(self.act(self.dw(F.pad(h, (1, 1, 1, 1), mode="reflect"))))

class M2V2(nn.Module):
    def __init__(self, channels=64, inner=16, hidden=16):
        super().__init__()
        self.pin = nn.Conv2d(channels, inner, 1, bias=False)
        self.color = _ExpertColor(inner)
        self.context = _ExpertContext(inner)
        self.detail = _ExpertDetail(inner)
        self.selector = nn.Sequential(
            nn.Linear(6 + inner, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.normal_(self.selector[-1].weight, std=0.01)
        nn.init.zeros_(self.selector[-1].bias)
        self.pout = nn.Conv2d(inner, channels, 1, bias=False)
        nn.init.zeros_(self.pout.weight)

    def forward(self, f, ctx):
        ctx = _match(ctx, f.shape[-2:])
        u = self.pin(f)
        t = _cluster_tokens(ctx["z"], ctx["w"], u)
        b = torch.softmax(self.selector(t), dim=-1)       # B,K,4
        e0 = self.color(u)
        e1 = self.context(u)
        e2 = ctx["r"] * self.detail(u)
        delta = 0
        for k in range(b.shape[1]):
            ck = (b[:, k, 0].view(-1, 1, 1, 1) * e0
                  + b[:, k, 1].view(-1, 1, 1, 1) * e1
                  + b[:, k, 2].view(-1, 1, 1, 1) * e2)
            delta = delta + ctx["w"][:, k:k + 1] * ck
        return f + 0.1 * self.pout(delta)

def _haar_kernels(device, dtype, channels):
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device, dtype=dtype)
    lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=device, dtype=dtype)
    hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], device=device, dtype=dtype)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=device, dtype=dtype)
    k = torch.stack([ll, lh, hl, hh], 0).unsqueeze(1).repeat(channels, 1, 1, 1)
    return k

def haar_dwt(x):
    c = x.shape[1]
    k = _haar_kernels(x.device, x.dtype, c)
    out = F.conv2d(x, k, stride=2, groups=c)
    return out.chunk(4, 1)

def haar_idwt(ll, lh, hl, hh):
    c = ll.shape[1]
    k = _haar_kernels(ll.device, ll.dtype, c)
    return F.conv_transpose2d(torch.cat([ll, lh, hl, hh], 1), k, stride=2, groups=c)



class M2V2R(nn.Module):
    """M2v2 with cluster-routing shrinkage rho.

    rho=1 reproduces M2v2. rho=0 uses the dark-mass weighted average selector.
    No trainable parameter is added; rho is a persistent buffer.
    """
    def __init__(self, channels=64, inner=16, hidden=16, rho=1.0):
        super().__init__()
        self.pin = nn.Conv2d(channels, inner, 1, bias=False)
        self.color = _ExpertColor(inner)
        self.context = _ExpertContext(inner)
        self.detail = _ExpertDetail(inner)
        self.selector = nn.Sequential(
            nn.Linear(6 + inner, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.normal_(self.selector[-1].weight, std=0.01)
        nn.init.zeros_(self.selector[-1].bias)
        self.pout = nn.Conv2d(inner, channels, 1, bias=False)
        nn.init.zeros_(self.pout.weight)
        self.register_buffer("rho", torch.tensor(float(rho)))

    def forward(self, f, ctx, dino_logits=None, dino_tokens=None, dino_cross=None,
                cluster_gate=None):
        ctx = _match(ctx, f.shape[-2:])
        if cluster_gate is not None:
            w = ctx["w"] * cluster_gate
            total = ctx["w"].sum(dim=1, keepdim=True)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(EPS) * total
            ctx = dict(ctx)
            ctx["w"] = w
        u = self.pin(f)
        t = _cluster_tokens(ctx["z"], ctx["w"], u)
        if dino_tokens is not None and dino_cross is not None:
            t = dino_cross(t, dino_tokens)
        logits = self.selector(t)
        if dino_logits is not None:
            logits = logits + dino_logits[:, None, :]
        b = torch.softmax(logits, dim=-1)       # B,K,4
        mass = ctx["w"].sum(dim=(2, 3)) + EPS  # B,K
        total = mass.sum(dim=1, keepdim=True)
        pi = mass / total
        bbar = (pi.unsqueeze(-1) * b).sum(dim=1, keepdim=True)  # B,1,4
        rho = self.rho.clamp(0.0, 1.0)
        bt = (1.0 - rho) * bbar + rho * b
        e0 = self.color(u)
        e1 = self.context(u)
        e2 = ctx["r"] * self.detail(u)
        delta = 0
        for k in range(bt.shape[1]):
            ck = (bt[:, k, 0].reshape(-1, 1, 1, 1) * e0
                  + bt[:, k, 1].reshape(-1, 1, 1, 1) * e1
                  + bt[:, k, 2].reshape(-1, 1, 1, 1) * e2)
            delta = delta + ctx["w"][:, k:k + 1] * ck
        return f + 0.1 * self.pout(delta)


class _SubbandExpert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 0, groups=c)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, h):
        h = F.pad(h, (1, 1, 1, 1), mode="reflect")
        return self.pw(self.act(self.dw(h)))

class M3V2(nn.Module):
    def __init__(self, channels=64, inner=16, hidden=16):
        super().__init__()
        self.pin = nn.Conv2d(channels, inner, 1, bias=False)
        self.mlp_ll = nn.Sequential(nn.Linear(6 + inner, hidden), nn.GELU(), nn.Linear(hidden, 2 * inner))
        self.mlp_tau = nn.Sequential(nn.Linear(6 + inner, hidden), nn.GELU(), nn.Linear(hidden, 3 * inner))
        self.mlp_h = nn.Sequential(nn.Linear(6 + inner, hidden), nn.GELU(), nn.Linear(hidden, 3 * inner))
        for mlp in (self.mlp_ll, self.mlp_tau, self.mlp_h):
            nn.init.normal_(mlp[-1].weight, std=0.01)
            nn.init.zeros_(mlp[-1].bias)
        with torch.no_grad():
            self.mlp_tau[-1].bias.fill_(-4.0)
        self.exp_h = nn.ModuleList([_SubbandExpert(inner) for _ in range(3)])
        self.pout = nn.Conv2d(inner, channels, 1, bias=False)
        nn.init.zeros_(self.pout.weight)
        self.register_buffer("tau_max", torch.tensor(0.02))

    def forward(self, f, ctx):
        ctx = _match(ctx, f.shape[-2:])
        u = self.pin(f)
        LL, LH, HL, HH = haar_dwt(u)
        hs = [LH, HL, HH]
        size = LL.shape[-2:]
        w_sub = F.avg_pool2d(ctx["w"], 2)
        z_sub = F.avg_pool2d(ctx["z"], 2)
        r_sub = F.avg_pool2d(ctx["r"], 2).clamp(0.0, 1.0)
        r5 = r_sub.unsqueeze(1)
        # tokens use LL statistics
        t = _cluster_tokens(z_sub, w_sub, LL)
        B, K, _ = t.shape
        gb = self.mlp_ll(t).view(B, K, 2, u.shape[1])
        gamma = 0.1 * torch.tanh(gb[:, :, 0]).unsqueeze(-1).unsqueeze(-1)
        beta = 0.1 * torch.tanh(gb[:, :, 1]).unsqueeze(-1).unsqueeze(-1)
        dLL = (w_sub.unsqueeze(2) * (gamma * LL.unsqueeze(1) + beta)).sum(1)
        tau = self.tau_max * torch.sigmoid(self.mlp_tau(t)).view(B, K, 3, u.shape[1])
        scale = 0.1 * torch.tanh(self.mlp_h(t)).view(B, K, 3, u.shape[1])
        dH = []
        for j, Hj in enumerate(hs):
            Hj5 = Hj.unsqueeze(1)
            tau_j = tau[:, :, j].unsqueeze(-1).unsqueeze(-1) * (1.0 - r5)
            Ht = torch.sign(Hj5) * F.relu(Hj5.abs() - tau_j)
            base = Ht - Hj5
            e = self.exp_h[j](Hj)
            cj = 0
            for k in range(K):
                ck = base[:, k] + r_sub * scale[:, k, j].unsqueeze(-1).unsqueeze(-1) * e
                cj = cj + w_sub[:, k:k + 1] * ck
            dH.append(cj)
        cand = haar_idwt(dLL, dH[0], dH[1], dH[2])
        return f + 0.1 * ctx["mask"] * self.pout(cand)

def build_dark_module_v2(variant, channels=64):
    if variant == "M1v2":
        return M1V2(channels)
    if variant == "M2v2":
        return M2V2(channels)
    if variant == "M2v2R":
        return M2V2R(channels)
    if variant == "M3v2":
        return M3V2(channels)
    raise ValueError(variant)
