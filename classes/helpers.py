import torch

def line_segment_intersection(p1, p2, p3, s):
    """
    p1 : (N,2) tensor, origine des rays
    p2 : (N,2) tensor, extrémité théorique des rays
    p3 : (M,2) tensor, origine des murs
    s  : (M,2) tensor, vecteurs murs (p4 − p3)
    ---
    renvoie : intersections (N,2) tensor, ou zeros si pas de hit
    """
    # Device
    device = p1.device

    # Ray direction
    r = p2 - p1                           # (N,2)
    # broadcast N×M
    r_b = r.unsqueeze(1)                 # (N,1,2)
    s_b = s.unsqueeze(0)                 # (1,M,2)
    p1_b = p1.unsqueeze(1)               # (N,1,2)
    p3_b = p3.unsqueeze(0)               # (1,M,2)

    # compute denominator
    denom = r_b[...,0]*s_b[...,1] - r_b[...,1]*s_b[...,0]    # (N,M)
    qp    = p3_b - p1_b                                      # (N,M,2)

    # compute parameters t and u
    t = (qp[...,0]*s_b[...,1] - qp[...,1]*s_b[...,0]) / denom
    u = (qp[...,0]*r_b[...,1] - qp[...,1]*r_b[...,0]) / denom

    # valid mask
    valid = (denom.abs() > 1e-8) & (t>=0) & (t<=1) & (u>=0) & (u<=1)  # (N,M)
    # replace invalid by +inf
    t_mask = torch.where(valid, t, torch.full_like(t, float('inf')))
    # smallest t per ray
    t_min, _ = t_mask.min(dim=1)    # (N,)

    # compute intersections
    inter = p1 + r * t_min.unsqueeze(1)   # (N,2)
    mask_hit = ~torch.isinf(t_min)
    # zero if no hit
    inter[~mask_hit] = 0.0

    return inter
