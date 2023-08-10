import pandas as pd
import numpy as np


def weight_diff(
    old_weights: pd.Series, new_weights: pd.Series, threshold=0.01, applied=False
):
    """
    Computes the difference between two weight dictionaries and returns a list of operations to perform to reach the new weights.

    Args:
        old_weights (dict): A dictionary of the old weights.
        new_weights (dict): A dictionary of the new weights.
        threshold (float, optional): The minimum percentage change to consider. Defaults to 0.01.
        applied (boolean, optional): If set to True, returns the new weights after applying operations

    Returns:
        list or dict: A list of tuples containing the key, operation, and percentage change for each operation to perform.
                      Or, if applied=True, a dict of adjusted new_weights.
    """
    if applied is True and threshold == 0:
        return new_weights
    # Get the set of all keys
    all_keys = set(list(old_weights.keys()) + list(new_weights.keys()))

    operations = []

    # Compute total absolute change
    total_change = sum(
        abs(new_weights.get(key, 0) - old_weights.get(key, 0)) for key in all_keys
    )

    adjusted_new_weights = new_weights.copy()  # Start with the original new_weights

    for key in all_keys:
        old_weight = old_weights.get(key, 0)
        new_weight = new_weights.get(key, 0)

        diff = new_weight - old_weight

        # Skip if absolute change is below threshold, adjusted for total_change
        abs_diff = abs(diff)
        if total_change <= 0 or abs_diff / total_change < threshold:
            if applied:  # Ignore this change if 'applied' is True
                adjusted_new_weights[key] = old_weight
            continue

        # Decide on the type of operation
        if old_weight >= 0:
            if diff > 0:
                operation = "Buy"
            else:  # old_weight >= 0 and diff < 0
                operation = "Sell"
        else:
            if diff < 0:
                operation = "Short"
            else:  # old_weight < 0 and diff > 0
                operation = "Cover"

        operations.append((key, operation, abs_diff / total_change))

    # Check if the sum of the percentages is somewhat close to 1
    assert np.isclose(adjusted_new_weights.sum(), 1, 0.1), (
        f"Sum of adjsuted weights is {adjusted_new_weights.sum()}, "
        f"but expected value is 1."
    )

    # Normalize the percentages so that they sum to 1.
    adjusted_new_weights = adjusted_new_weights / adjusted_new_weights.sum()

    return adjusted_new_weights if applied else operations
