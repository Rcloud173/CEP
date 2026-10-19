"""
Acoustic Feature Extraction Module
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

This module extracts 14 statistical time-domain and frequency-domain features
from 1-second acoustic segments to feed into machine learning classifiers.

Feature Set:
1. RMS (Root Mean Square Energy)
2. Mean amplitude
3. Variance
4. Standard deviation
5. Peak amplitude
6. Zero Crossing Rate (ZCR)
7. Dominant Frequency (via FFT)
8. Spectral Centroid (brightness / center of mass of spectrum)
9. Spectral Bandwidth (spread around centroid)
10. Spectral Roll-off (frequency containing 85% of total energy)
11. Band Energy 1 (0 - 500 Hz: Flow friction and pump rumble)
12. Band Energy 2 (500 - 1500 Hz: Primary water leak hiss band)
13. Band Energy 3 (1500 - 2500 Hz: Secondary leak turbulence band)
14. Band Energy 4 (2500 - 4000 Hz: Upper acoustic boundary)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path regardless of how script is executed
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

import numpy as np
import pandas as pd
import librosa
from preprocessing.preprocess import (
    load_signal,
    validate_signal,
    bandpass_filter,
    normalize,
    segment,
)

# Standard list of 14 feature column names
FEATURE_NAMES: List[str] = [
    "rms",
    "mean",
    "variance",
    "std",
    "peak",
    "zcr",
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "band_energy_0_500",
    "band_energy_500_1500",
    "band_energy_1500_2500",
    "band_energy_2500_4000",
]


def extract_features_from_segment(
    seg: np.ndarray,
    sr: int = 8000,
) -> Dict[str, float]:
    """
    Extract 14 time and frequency domain acoustic features from a 1-second segment.

    Parameters:
    - seg: 1D NumPy array representing one segment (typically 8000 samples).
    - sr: Sampling rate in Hz (default: 8000 Hz).

    Returns:
    - Dictionary mapping feature name to float value.
    """
    seg = np.asarray(seg, dtype=np.float32)
    n = len(seg)

    # ---------------------------------------------------------
    # 1. TIME-DOMAIN STATISTICAL FEATURES
    # ---------------------------------------------------------
    rms_val = float(np.sqrt(np.mean(seg**2)))
    mean_val = float(np.mean(seg))
    var_val = float(np.var(seg))
    std_val = float(np.std(seg))
    peak_val = float(np.max(np.abs(seg)))

    # Zero-crossing rate: fraction of sign changes in time
    zcr_raw = librosa.feature.zero_crossing_rate(seg, frame_length=n, hop_length=n + 1)
    zcr_val = float(zcr_raw[0, 0])

    # ---------------------------------------------------------
    # 2. FREQUENCY-DOMAIN FFT & SPECTRAL SHAPE FEATURES
    # ---------------------------------------------------------
    # Real Fast Fourier Transform (RFFT)
    fft_vals = np.fft.rfft(seg)
    fft_freqs = np.fft.rfftfreq(n, 1.0 / sr)
    magnitude_spectrum = np.abs(fft_vals)
    power_spectrum = magnitude_spectrum**2

    # Dominant Frequency (Frequency bin with maximum magnitude)
    dominant_freq = float(fft_freqs[np.argmax(magnitude_spectrum)])

    # Spectral Centroid (Center of spectral energy distribution)
    cent = librosa.feature.spectral_centroid(y=seg, sr=sr)
    spectral_centroid = float(np.mean(cent))

    # Spectral Bandwidth (Spread around the spectral centroid)
    bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
    spectral_bandwidth = float(np.mean(bw))

    # Spectral Roll-off (Frequency below which 85% of spectral energy lies)
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)
    spectral_rolloff = float(np.mean(rolloff))

    # ---------------------------------------------------------
    # 3. 4 FREQUENCY-BAND ENERGIES
    # ---------------------------------------------------------
    # Band 1: 0 to 500 Hz (Normal background flow, pump rumble)
    mask_b1 = (fft_freqs >= 0.0) & (fft_freqs < 500.0)
    band_energy_1 = float(np.sum(power_spectrum[mask_b1]))

    # Band 2: 500 to 1500 Hz (Turbulent acoustic jet / orifice escape)
    mask_b2 = (fft_freqs >= 500.0) & (fft_freqs < 1500.0)
    band_energy_2 = float(np.sum(power_spectrum[mask_b2]))

    # Band 3: 1500 to 2500 Hz (High-pressure acoustic resonance)
    mask_b3 = (fft_freqs >= 1500.0) & (fft_freqs < 2500.0)
    band_energy_3 = float(np.sum(power_spectrum[mask_b3]))

    # Band 4: 2500 to 4000 Hz (High frequency harmonics)
    mask_b4 = (fft_freqs >= 2500.0) & (fft_freqs <= 4000.0)
    band_energy_4 = float(np.sum(power_spectrum[mask_b4]))

    return {
        "rms": rms_val,
        "mean": mean_val,
        "variance": var_val,
        "std": std_val,
        "peak": peak_val,
        "zcr": zcr_val,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "band_energy_0_500": band_energy_1,
        "band_energy_500_1500": band_energy_2,
        "band_energy_1500_2500": band_energy_3,
        "band_energy_2500_4000": band_energy_4,
    }


def extract_features_from_signal(
    sig: np.ndarray,
    sr: int = 8000,
    file_id: str = "custom_signal",
    label: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process a continuous audio signal:
    1. Validate
    2. Bandpass filter (100 - 3000 Hz)
    3. Normalize
    4. Segment (1-second windows)
    5. Extract 14 features per segment

    Returns:
    - Pandas DataFrame where each row corresponds to one 1-second segment.
    """
    validate_signal(sig, sr=sr, min_duration=1.0, raise_error=True)
    filtered = bandpass_filter(sig, sr=sr)
    normalized = normalize(filtered)
    segments = segment(normalized, sr=sr, window_sec=1.0, hop_sec=1.0)

    rows = []
    for seg_idx, seg in enumerate(segments):
        feat_dict = extract_features_from_segment(seg, sr=sr)
        row = {
            "file_id": file_id,
            "segment_id": seg_idx,
            **feat_dict,
        }
        if label is not None:
            row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


def extract_dataset_features(
    dataset_dir: str,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Iterate over normal/ and leak/ directories, preprocess every WAV file,
    extract features for all 1-second segments, and compile into a single DataFrame.

    Label encoding:
    - 0 = NORMAL CONDITION
    - 1 = POSSIBLE WATER LEAK
    """
    normal_dir = os.path.join(dataset_dir, "normal")
    leak_dir = os.path.join(dataset_dir, "leak")

    all_rows = []

    classes = [
        ("normal", normal_dir, 0),
        ("leak", leak_dir, 1),
    ]

    for class_name, folder, label in classes:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset subfolder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        print(f"Extracting features from {len(files)} {class_name} files...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                sig, sr = load_signal(fpath, target_sr=8000)
                df_file = extract_features_from_signal(
                    sig, sr=sr, file_id=fname, label=label
                )
                all_rows.append(df_file)
            except Exception as e:
                print(f"Warning: Failed to process {fname}: {e}")

    dataset_df = pd.concat(all_rows, ignore_index=True)

    if output_csv:
        dataset_df.to_csv(output_csv, index=False)
        print(f"Saved dataset features to: {output_csv}")

    return dataset_df


if __name__ == "__main__":
    # Test feature extraction on the generated dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset")
    csv_path = os.path.join(dataset_path, "extracted_features.csv")

    print("=" * 60)
    print("STEP 3: EXTRACTING FEATURES FROM DATASET")
    print("=" * 60)
    df = extract_dataset_features(dataset_path, output_csv=csv_path)
    print(f"Extraction complete! Total segments extracted: {len(df)}")
    print(f"Unique files: {df['file_id'].nunique()}")
    print(f"Class distribution:\n{df['label'].value_counts().to_dict()}")
    print("\nFirst 3 rows:")
    print(df[["file_id", "segment_id", "rms", "dominant_freq", "spectral_centroid", "label"]].head(3))
    print("=" * 60)
