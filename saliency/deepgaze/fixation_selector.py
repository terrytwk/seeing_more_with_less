import numpy as np


def select_fixation_points(saliency_map, num_fixations=1, min_distance=64, threshold=None):
    """Select well-spaced fixation points from a saliency probability map.

    Returns dictionaries with centroids in the existing JSON convention:
    centroid is [y, x], while filter_image.py converts that back to [x, y].
    """
    if saliency_map.ndim != 2:
        raise ValueError("saliency_map must be a 2D array")
    if num_fixations < 1:
        raise ValueError("num_fixations must be at least 1")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")

    scores = np.asarray(saliency_map, dtype=np.float64).copy()
    scores[~np.isfinite(scores)] = -np.inf

    selected = []
    yy, xx = np.ogrid[:scores.shape[0], :scores.shape[1]]

    for obj_id in range(num_fixations):
        flat_index = int(np.argmax(scores))
        best_score = float(scores.flat[flat_index])
        if not np.isfinite(best_score):
            break
        if threshold is not None and best_score < threshold:
            break

        y, x = np.unravel_index(flat_index, scores.shape)
        selected.append({
            "obj_id": obj_id,
            "centroid": [int(y), int(x)],
            "score": best_score,
        })

        if min_distance == 0:
            scores[y, x] = -np.inf
        else:
            suppression_mask = (yy - y) ** 2 + (xx - x) ** 2 <= min_distance ** 2
            scores[suppression_mask] = -np.inf

    return selected

