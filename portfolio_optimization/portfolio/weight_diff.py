def weight_diff(old_weights, new_weights, threshold=0.01):
    """
    Computes the difference between two weight dictionaries and returns a list of operations to perform to reach the new weights.

    Args:
        old_weights (dict): A dictionary of the old weights.
        new_weights (dict): A dictionary of the new weights.
        threshold (float, optional): The minimum percentage change to consider. Defaults to 0.01.

    Returns:
        list: A list of tuples containing the key, operation, and percentage change for each operation to perform.
    """
    # Get the set of all keys
    all_keys = set(list(old_weights.keys()) + list(new_weights.keys()))

    operations = []

    # Compute total absolute change
    total_change = sum(
        abs(new_weights.get(key, 0) - old_weights.get(key, 0)) for key in all_keys
    )

    for key in all_keys:
        old_weight = old_weights.get(key, 0)
        new_weight = new_weights.get(key, 0)

        diff = new_weight - old_weight

        # Skip if absolute change is below threshold, adjusted for total_change
        abs_diff = abs(diff)
        if abs_diff / total_change < threshold:
            continue

        # Decide on the type of operation
        if old_weight >= 0:
            if diff > 0:
                operation = "Buy"
            else:  # old_weight >= 0 and diff < 0
                operation = "Sell"
        elif old_weight < 0:
            if diff < 0:
                operation = "Short"
            else:  # old_weight < 0 and diff > 0
                operation = "Cover"

        operations.append((key, operation, abs_diff / total_change))

    return operations
