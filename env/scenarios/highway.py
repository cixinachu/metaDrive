"""Formal procedural Highway research scenarios and optional auxiliary maps."""
HIGHWAY_SCENARIOS = {
    "straight": {"label": "Highway Straight", "map": "S", "role": "train"},
    "curve": {"label": "Highway Curve", "map": "C", "role": "transfer"},
    "merge": {"label": "Highway Merge", "map": "r", "role": "transfer"},
}
AUXILIARY_SCENARIOS = {
    "intersection": {"label": "Intersection", "map": "X"},
    "t_junction": {"label": "T-junction", "map": "T"},
    "roundabout": {"label": "Roundabout", "map": "O"},
}
