from __future__ import annotations


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Even/odd ray casting; polygon points are top-down feet."""
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def zone_at(x: float, y: float, zones: dict) -> str | None:
    for name, zone in zones.items():
        if zone.get("type", "").endswith("_roi"):
            continue
        if point_in_polygon(x, y, zone.get("points", [])):
            return name
    return None
