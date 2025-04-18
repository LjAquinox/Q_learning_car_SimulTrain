import pygame

# --- Helper Functions ---
def line_segment_intersection(p1, p2, p3, p4):
    """ Finds the intersection point between segments [p1, p2] and [p3, p4].
        Returns the intersection point (Vector2) or None if no intersection.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = pygame.Vector2(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return intersection_point
    return None