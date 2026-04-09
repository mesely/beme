DEFAULT_BOUNTY_MAP = {1: 1.5, 2: 2.0, 7: 1.5, 8: 2.0}


def get_bounty(bounty_map, class_index):
    """
    Return the bounty multiplier beta_c for a given class.
    Returns 1.0 if class is not in the map or map is disabled.
    """
    if not bounty_map:
        return 1.0
    return bounty_map.get(class_index, 1.0)
