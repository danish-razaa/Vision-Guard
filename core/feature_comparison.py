"""Feature-vector comparison helpers for controlled experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from app.xai import feature_domain


def compare_feature_vectors(
    original_features: Mapping[str, float],
    modified_features: Mapping[str, float],
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    """Rank shared finite features by absolute numerical change."""
    rows = []
    for name in original_features.keys() & modified_features.keys():
        original = float(original_features[name])
        modified = float(modified_features[name])
        if not np.isfinite(original) or not np.isfinite(modified):
            continue
        signed_delta = modified - original
        delta = abs(signed_delta)
        relative = delta / abs(original) if abs(original) > 1e-12 else (0.0 if delta == 0 else None)
        rows.append({
            "feature_name": name,
            "original_value": original,
            "modified_value": modified,
            "absolute_delta": delta,
            "signed_delta": signed_delta,
            "relative_delta": relative,
            "feature_domain": feature_domain(name),
        })
    return sorted(rows, key=lambda item: float(item["absolute_delta"]), reverse=True)[:top_k]
