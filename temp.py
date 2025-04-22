import numpy as np

# --- Helper Functions ---
def line_segment_intersection(p1, p2, p3, p4): 
    """
    Finds the nearest intersection point between each segment [p1[i], p2[i]] and any of the segments [p3[j], p4[j]].
    Returns an array of shape (num_rays, 2), where each row is the intersection point (x, y), or [0, 0] if no intersection.

    Parameters
    ----------
    p1 : np.ndarray, shape (N, 2)
        Start points of the N rays/segments.
    p2 : np.ndarray, shape (N, 2)
        End points of the N rays/segments.
    p3 : np.ndarray, shape (M, 2)
        Start points of the M wall segments.
    p4 : np.ndarray, shape (M, 2)
        End points of the M wall segments.

    Returns
    -------
    intersections : np.ndarray, shape (N, 2)
        Intersection points for each ray; if no intersection, returns [0, 0].
    """
    N = p1.shape[0]
    # Direction vectors for rays and walls
    r = p2 - p1                # shape (N, 2)
    s = p4 - p3                # shape (M, 2)

    # Broadcast to pairwise combinations
    r_b = r[:, None, :]        # shape (N, M, 2)
    s_b = s[None, :, :]        # shape (N, M, 2)
    p1_b = p1[:, None, :]      # shape (N, 1, 2)
    p3_b = p3[None, :, :]      # shape (1, M, 2)

    def cross2(a, b):
        """2D cross product a x b for arrays"""
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # Denominator for intersection formulas
    denom = cross2(r_b, s_b)   # shape (N, M)
    qp = p3_b - p1_b           # shape (N, M, 2)

    # Compute parameters t and u safely, suppress warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross2(qp, s_b) / denom  # shape (N, M)
        u = cross2(qp, r_b) / denom  # shape (N, M)

    # Valid intersection mask: parallel (denom=0) excluded, and 0<=t,u<=1
    valid = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    # For invalid or non-intersecting, set t to infinity so it won't be chosen
    t_masked = np.where(valid, t, np.inf)
    # Smallest t per ray (closest hit) or inf if none
    t_min = np.min(t_masked, axis=1)  # shape (N,)

    # Prepare output array of zeros
    intersections = np.zeros((N, 2), dtype=float)

    # For rays that intersect, compute intersection = p1 + t_min * r
    hit_mask = ~np.isinf(t_min)
    if np.any(hit_mask):
        intersections[hit_mask] = (
            p1[hit_mask] + 
            (r[hit_mask] * t_min[hit_mask][:, None])
        )

    return intersections