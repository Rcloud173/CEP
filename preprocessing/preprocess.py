"""
Signal Preprocessing Module for Acoustic Water Leak Detection
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

Key Stages:
1. Load Signal: Read WAV file and convert to mono float32 at target sampling rate.
2. Validate Signal: Check that signal is not empty, silent, or too short.
3. Bandpass Filter: Butterworth filter (100 Hz - 3000 Hz) to isolate acoustic leak frequencies.
4. Normalize: Scale amplitude to [-1.0, +1.0] for uniform feature comparison.
5. Segment: Divide continuous audio into 1-second windows for localized analysis.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
from scipy import signal
import soundfile as sf
import librosa


def load_signal(file_path: str, target_sr: int = 8000) -> Tuple[np.ndarray, int]:
    """
    Load an audio signal from a .wav file.

    Parameters:
    - file_path: Path to the audio file.
    - target_sr: Desired sampling rate in Hz (default: 8000 Hz).

    Returns:
    - (sig, sr): Mono float32 NumPy array and the sampling rate.

    Raises:
    - FileNotFoundError: If file does not exist.
    - ValueError: If file cannot be decoded as audio.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Load audio using soundfile
        data, sr = sf.read(file_path, dtype="float32")
    except Exception as e:
        # Fallback to librosa if soundfile fails
        try:
            data, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception as inner_e:
            raise ValueError(f"Unable to read audio file '{file_path}': {e} / {inner_e}")

    # Convert multi-channel (stereo) to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1 if data.shape[0] > data.shape[1] else 0)

    data = np.ascontiguousarray(data, dtype=np.float32)

    # Resample to target sampling rate if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return data, sr


def validate_signal(
    sig: np.ndarray,
    sr: int = 8000,
    min_duration: float = 1.0,
    raise_error: bool = True,
) -> Tuple[bool, str]:
    """
    Validate acoustic signal quality and requirements.

    Checks:
    - Input is non-empty 1D array.
    - No NaN or Inf values.
    - Duration >= min_duration.
    - Signal is not completely silent / zero amplitude.

    Parameters:
    - sig: 1D NumPy array representing audio samples.
    - sr: Sampling rate in Hz.
    - min_duration: Minimum required signal duration in seconds (default: 1.0s).
    - raise_error: If True, raises ValueError upon failure. If False, returns (False, error_msg).

    Returns:
    - (is_valid, message): Boolean and status explanation.
    """
    if sig is None or not isinstance(sig, (np.ndarray, list)) or len(sig) == 0:
        msg = "Signal is empty or invalid data type."
        if raise_error:
            raise ValueError(msg)
        return False, msg

    sig_arr = np.asarray(sig)

    if not np.all(np.isfinite(sig_arr)):
        msg = "Signal contains invalid numerical values (NaN or Infinity)."
        if raise_error:
            raise ValueError(msg)
        return False, msg

    duration = len(sig_arr) / float(sr)
    if duration < min_duration:
        msg = f"Signal duration ({duration:.2f}s) is shorter than minimum required ({min_duration}s)."
        if raise_error:
            raise ValueError(msg)
        return False, msg

    # Check for complete silence (RMS < 1e-5 or peak < 1e-4)
    rms = float(np.sqrt(np.mean(sig_arr**2)))
    peak = float(np.max(np.abs(sig_arr)))
    if rms < 1e-5 or peak < 1e-4:
        msg = f"Signal is silent or near-zero amplitude (RMS={rms:.2e}, Peak={peak:.2e})."
        if raise_error:
            raise ValueError(msg)
        return False, msg

    return True, "Signal validation passed successfully."


def bandpass_filter(
    sig: np.ndarray,
    sr: int = 8000,
    lowcut: float = 100.0,
    highcut: float = 3000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Viva Explanation:
    - Municipal pipe leaks produce vibrations predominantly between 500 Hz - 3000 Hz.
    - Frequencies below 100 Hz contain electrical 50 Hz hum and slow structural vibrations.
    - Frequencies above 3000 Hz are mostly high-frequency ambient noise.
    - Filtering removes noise outside this band, boosting detection accuracy.

    Parameters:
    - sig: Audio signal array.
    - sr: Sampling rate in Hz.
    - lowcut: Lower cutoff frequency (default: 100 Hz).
    - highcut: Upper cutoff frequency (default: 3000 Hz).
    - order: Butterworth filter order (default: 4).

    Returns:
    - Filtered signal array.
    """
    nyquist = 0.5 * sr
    low = max(10.0, lowcut) / nyquist
    high = min(highcut, nyquist - 10.0) / nyquist

    # Second-order sections (SOS) representation for numerical stability
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered.astype(np.float32)


def normalize(sig: np.ndarray) -> np.ndarray:
    """
    Peak-normalize the acoustic signal to the dynamic range [-1.0, +1.0].

    Formula:
    normalized_sig = sig / max(|sig|)

    This eliminates differences caused solely by recording distance or microphone gain.
    """
    peak = np.max(np.abs(sig))
    if peak > 1e-9:
        normalized = sig / peak
    else:
        normalized = sig.copy()
    return normalized.astype(np.float32)


def segment(
    sig: np.ndarray,
    sr: int = 8000,
    window_sec: float = 1.0,
    hop_sec: float = 1.0,
) -> List[np.ndarray]:
    """
    Slice the continuous audio signal into fixed-length windows.

    Viva Explanation:
    - Acoustic events change over time. Slicing into 1.0-second windows allows
      the system to compute localized statistical features and detect short bursts.
    - A 5-second audio yields 5 independent segments of 8,000 samples each.

    Parameters:
    - sig: Continuous audio signal array.
    - sr: Sampling rate in Hz.
    - window_sec: Length of each segment in seconds (default: 1.0s).
    - hop_sec: Step size between consecutive segments in seconds (default: 1.0s).

    Returns:
    - List of 1D NumPy arrays, each containing window_samples samples.
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    segments = []
    n_samples = len(sig)

    for start in range(0, n_samples - window_samples + 1, hop_samples):
        chunk = sig[start : start + window_samples]
        segments.append(chunk)

    return segments
