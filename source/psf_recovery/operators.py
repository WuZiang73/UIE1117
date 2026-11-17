import torch
import torch.nn.functional as F


def _conv3(x, k):
    # x: B,3,H,W; k: 4,1,kh,kw -> B,4,3,H,W
    b, c, h, w = x.shape
    r = k.shape[-1] // 2
    xp = F.pad(x, (r, r, r, r), mode="reflect")
    m = k.shape[0]
    # grouped conv: group c uses kernels m=0..M-1 in order c*M + m
    ww = k[:, None].repeat(1, c, 1, 1).permute(1, 0, 2, 3).reshape(c * m, 1, k.shape[-2], k.shape[-1])
    out = F.conv2d(xp, ww, groups=c)
    return out.view(b, c, m, h, w)


def render_baseline(j, kernels, coeff, bias):
    # coeff B,3,4; kernels 4,kh,kw; j B,3,H,W
    blurred = _conv3(j, kernels)
    return (blurred * coeff[:, :, :, None, None]).sum(2) + bias


def apply_correction(d, kernels, coeff):
    # linear A; d: B,3,H,W; zero outside the patch
    b, c, h, w = d.shape
    blurred = _conv3(d, kernels)
    return (blurred * coeff[:, :, :, None, None]).sum(2)


def adjoint_correction(d, kernels, coeff, v):
    # exact VJP of apply_correction at d
    with torch.enable_grad():
        dd = d.detach().requires_grad_(True)
        y = apply_correction(dd, kernels, coeff)
        g = torch.autograd.grad(y, dd, v, create_graph=False, retain_graph=False)[0]
    return g.detach()


def cg_solve(A, At, b, lam=0.01, steps=5):
    x = torch.zeros_like(b)
    r = b - A(x)
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(steps):
        Ap = A(p) + lam * p
        denom = (p * Ap).sum()
        if denom.abs() < 1e-12:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.abs() < 1e-18:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x
