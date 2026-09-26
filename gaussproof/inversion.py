"""Pixel reconstruction from estimated clipped gradients, with no target pixels."""
import numpy as np
import torch


def differentiable_gradient(model, image, label, clip):
    loss=torch.nn.functional.cross_entropy(model(image),torch.tensor([int(label)]))
    values=torch.autograd.grad(loss,tuple(model.parameters()),create_graph=True)
    g=torch.cat([p.flatten() for p in values])
    return g*torch.clamp(clip/g.norm().clamp_min(1e-12),max=1)


def invert(models, estimated_gradient, label, clip=1., steps=180, restarts=2, seed=0):
    target=torch.as_tensor(estimated_gradient,dtype=torch.float32)
    if target.ndim!=1 or not torch.isfinite(target).all():raise ValueError('Finite flat gradient required')
    best_score=float('inf');best_image=None
    for restart in range(restarts):
        rng=torch.Generator().manual_seed(seed+restart)
        pixels=(.25+.5*torch.rand((1,1,28,28),generator=rng)).requires_grad_()
        opt=torch.optim.Adam([pixels],lr=.06)
        for _ in range(steps):
            g=torch.cat([differentiable_gradient(model,pixels,label,clip) for model in models])
            mismatch=(g-target).square().sum()/target.square().sum().clamp_min(1e-6)
            tv=(pixels[:,:,:,1:]-pixels[:,:,:,:-1]).abs().mean()+(pixels[:,:,1:,:]-pixels[:,:,:-1,:]).abs().mean()
            objective=mismatch+1e-4*tv
            opt.zero_grad()
            derivative,=torch.autograd.grad(objective,pixels)
            pixels.grad=derivative
            opt.step()
            with torch.no_grad():pixels.clamp_(0,1)
        final=torch.cat([differentiable_gradient(model,pixels,label,clip) for model in models])
        score=float(((final-target).square().sum()/target.square().sum().clamp_min(1e-6)).detach())
        if score<best_score:best_score=score;best_image=pixels.detach().numpy()[0,0].copy()
    return best_image,best_score


def image_metrics(reconstruction, original):
    mse=float(np.mean((np.asarray(reconstruction)-np.asarray(original))**2))
    return dict(pixel_mse=mse,psnr_db=float(-10*np.log10(max(mse,1e-12))))
