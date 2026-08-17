import numpy as np
from scipy import stats
from typing import Dict


def calculate_time_domain_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """Calculates comprehensive time-domain statistical metrics for a 1D signal vector."""
    if len(signal) == 0:
        return {}

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    var_val = float(np.var(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    ptp_val = max_val - min_val

    # Root Mean Square (RMS)
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    # Skewness & Kurtosis
    skew_val = float(stats.skew(signal)) if len(signal) > 2 else 0.0
    kurt_val = float(stats.kurtosis(signal)) if len(signal) > 3 else 0.0

    # Crest Factor = Peak / RMS (with safety division)
    peak_val = max(abs(min_val), abs(max_val))
    crest_factor = float(peak_val / (rms_val + 1e-8))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_var": var_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_ptp": ptp_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurtosis": kurt_val,
        f"{prefix}_crest_factor": crest_factor,
    }
