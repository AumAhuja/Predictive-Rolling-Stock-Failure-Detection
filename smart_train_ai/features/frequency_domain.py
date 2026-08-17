import numpy as np
from typing import Dict


def calculate_frequency_domain_features(
    signal: np.ndarray,
    prefix: str,
    sampling_hz: float = 100.0
) -> Dict[str, float]:
    """Calculates FFT-based frequency-domain spectral features for a 1D signal vector."""
    if len(signal) < 4:
        return {
            f"{prefix}_dominant_freq": 0.0,
            f"{prefix}_spectral_energy": 0.0,
            f"{prefix}_spectral_centroid": 0.0,
            f"{prefix}_band_energy_low": 0.0,
            f"{prefix}_band_energy_mid": 0.0,
            f"{prefix}_band_energy_high": 0.0,
        }

    # Remove DC component for spectral analysis
    sig_detrended = signal - np.mean(signal)

    # Compute Real FFT
    n = len(sig_detrended)
    fft_vals = np.fft.rfft(sig_detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_hz)

    magnitude = np.abs(fft_vals)
    power_spectrum = magnitude ** 2

    # Dominant Frequency
    max_idx = np.argmax(magnitude[1:]) + 1 if len(magnitude) > 1 else 0
    dominant_freq = float(fft_freqs[max_idx]) if len(fft_freqs) > max_idx else 0.0

    # Total Spectral Energy
    spectral_energy = float(np.sum(power_spectrum))

    # Spectral Centroid (weighted mean frequency)
    sum_power = np.sum(power_spectrum)
    if sum_power > 1e-10:
        spectral_centroid = float(np.sum(fft_freqs * power_spectrum) / sum_power)
    else:
        spectral_centroid = 0.0

    # Sub-band Energies: Low (0-10Hz), Mid (10-30Hz), High (30-50Hz)
    low_mask = (fft_freqs >= 0.0) & (fft_freqs < 10.0)
    mid_mask = (fft_freqs >= 10.0) & (fft_freqs < 30.0)
    high_mask = (fft_freqs >= 30.0) & (fft_freqs <= 50.0)

    band_energy_low = float(np.sum(power_spectrum[low_mask]))
    band_energy_mid = float(np.sum(power_spectrum[mid_mask]))
    band_energy_high = float(np.sum(power_spectrum[high_mask]))

    return {
        f"{prefix}_dominant_freq": dominant_freq,
        f"{prefix}_spectral_energy": spectral_energy,
        f"{prefix}_spectral_centroid": spectral_centroid,
        f"{prefix}_band_energy_low": band_energy_low,
        f"{prefix}_band_energy_mid": band_energy_mid,
        f"{prefix}_band_energy_high": band_energy_high,
    }
