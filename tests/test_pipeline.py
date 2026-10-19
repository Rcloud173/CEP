"""
Unit and Integration Test Suite for Water Leak Detection Pipeline
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals

Test Cases (pytest):
1. test_valid_signal_accepted
2. test_invalid_file_rejected
3. test_empty_signal_rejected
4. test_silent_signal_rejected
5. test_too_short_signal_rejected
6. test_filter_and_normalization
7. test_feature_extraction_completeness
8. test_normal_sample_classified_as_normal
9. test_leak_sample_classified_as_leak
10. test_noisy_signal_processed_gracefully
11. test_database_save_and_retrieve
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PROJECT_DIR not in sys.path:
    sys.path.insert(0, BASE_PROJECT_DIR)

from preprocessing.preprocess import (
    load_signal,
    validate_signal,
    bandpass_filter,
    normalize,
    segment,
)
from feature_extraction.features import (
    extract_features_from_segment,
    extract_features_from_signal,
    FEATURE_NAMES,
)
from application.db import init_db, save_result, get_history, clear_history


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(scope="session")
def trained_model():
    """Load the trained machine learning pipeline artifact."""
    model_path = os.path.join(BASE_PROJECT_DIR, "model", "leak_model.joblib")
    if not os.path.exists(model_path):
        pytest.fail(f"Trained model not found at {model_path}. Run train.py first!")
    artifact = joblib.load(model_path)
    return artifact["best_model"]


@pytest.fixture
def synthetic_normal_signal():
    """Create a 5.0s synthetic normal pipeline signal (40,000 samples at 8 kHz)."""
    sr = 8000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    # 50 Hz hum + low frequency rumble + pinkish noise
    hum = 0.2 * np.sin(2 * np.pi * 50.0 * t)
    rumble = 0.3 * np.sin(2 * np.pi * 25.0 * t)
    noise = 0.1 * np.random.RandomState(42).randn(len(t))
    sig = hum + rumble + noise
    return sig.astype(np.float32), sr


@pytest.fixture
def synthetic_leak_signal():
    """Create a 5.0s synthetic leak acoustic signal with 1200 Hz hiss."""
    from scipy import signal as sp_signal
    sr = 8000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    # Broadband hiss bandpassed 900-1500 Hz
    white = np.random.RandomState(42).randn(len(t))
    sos = sp_signal.butter(4, [900, 1500], btype="bandpass", fs=sr, output="sos")
    hiss = sp_signal.sosfiltfilt(sos, white) * 0.9
    hum = 0.15 * np.sin(2 * np.pi * 50.0 * t)
    sig = hum + hiss
    return sig.astype(np.float32), sr


# ==========================================
# TESTS
# ==========================================
def test_valid_signal_accepted(synthetic_normal_signal):
    """TC01: Valid signal of proper length and amplitude must pass validation."""
    sig, sr = synthetic_normal_signal
    is_valid, msg = validate_signal(sig, sr=sr, min_duration=1.0, raise_error=False)
    assert is_valid is True
    assert "passed" in msg.lower()


def test_invalid_file_rejected():
    """TC02: Non-existent audio file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_signal("non_existent_path_to_audio_12345.wav")


def test_empty_signal_rejected():
    """TC03: Empty signal array must be rejected by validation."""
    empty_sig = np.array([], dtype=np.float32)
    with pytest.raises(ValueError):
        validate_signal(empty_sig, sr=8000, raise_error=True)

    is_valid, msg = validate_signal(empty_sig, sr=8000, raise_error=False)
    assert is_valid is False


def test_silent_signal_rejected():
    """TC04: Near-zero / completely silent signal must be rejected."""
    silent_sig = np.zeros(8000 * 3, dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        validate_signal(silent_sig, sr=8000, raise_error=True)
    assert "silent" in str(excinfo.value).lower()


def test_too_short_signal_rejected():
    """TC05: Audio shorter than 1.0 second minimum requirement must be rejected."""
    short_sig = np.ones(4000, dtype=np.float32) * 0.1  # 0.5s at 8000 Hz
    with pytest.raises(ValueError) as excinfo:
        validate_signal(short_sig, sr=8000, min_duration=1.0, raise_error=True)
    assert "shorter" in str(excinfo.value).lower()


def test_filter_and_normalization(synthetic_normal_signal):
    """TC06: Bandpass filtering and normalization must output clean [-1, 1] signal without NaNs."""
    sig, sr = synthetic_normal_signal
    filtered = bandpass_filter(sig, sr=sr, lowcut=100.0, highcut=3000.0)
    normalized = normalize(filtered)

    assert normalized.shape == sig.shape
    assert np.all(np.isfinite(normalized))
    assert np.max(np.abs(normalized)) <= 1.0001
    assert np.max(np.abs(normalized)) >= 0.99  # Peak normalized to 1.0


def test_feature_extraction_completeness(synthetic_normal_signal):
    """TC07: Feature extraction must return all 14 designated features per 1-second segment."""
    sig, sr = synthetic_normal_signal
    df = extract_features_from_signal(sig, sr=sr, file_id="test_sample.wav")

    assert len(df) == 5  # 5 segments for 5-second audio
    for feat_col in FEATURE_NAMES:
        assert feat_col in df.columns, f"Missing feature column: {feat_col}"
        assert not df[feat_col].isnull().any(), f"NaN found in feature: {feat_col}"


def test_normal_sample_classified_as_normal(trained_model):
    """TC08: A baseline normal pipeline WAV file must be classified as NORMAL (0)."""
    normal_wav = os.path.join(BASE_PROJECT_DIR, "dataset", "normal", "normal_001.wav")
    if not os.path.exists(normal_wav):
        pytest.skip("normal_001.wav not found. Run generate_synthetic.py first.")

    sig, sr = load_signal(normal_wav, target_sr=8000)
    df_feat = extract_features_from_signal(sig, sr=sr)
    X = df_feat[FEATURE_NAMES].values

    segment_preds = trained_model.predict(X)
    probs = trained_model.predict_proba(X)[:, 1]
    mean_leak_prob = np.mean(probs)

    assert mean_leak_prob < 0.50, f"Expected Normal, but mean leak prob was {mean_leak_prob:.3f}"
    # Majority of segments should be 0
    assert np.sum(segment_preds == 0) >= 3


def test_leak_sample_classified_as_leak(trained_model):
    """TC09: A representative leak acoustic WAV file must be classified as POSSIBLE LEAK (1)."""
    leak_wav = os.path.join(BASE_PROJECT_DIR, "dataset", "leak", "leak_001.wav")
    if not os.path.exists(leak_wav):
        pytest.skip("leak_001.wav not found. Run generate_synthetic.py first.")

    sig, sr = load_signal(leak_wav, target_sr=8000)
    df_feat = extract_features_from_signal(sig, sr=sr)
    X = df_feat[FEATURE_NAMES].values

    segment_preds = trained_model.predict(X)
    probs = trained_model.predict_proba(X)[:, 1]
    mean_leak_prob = np.mean(probs)

    assert mean_leak_prob >= 0.50, f"Expected Leak, but mean leak prob was {mean_leak_prob:.3f}"
    # Majority of segments should be 1
    assert np.sum(segment_preds == 1) >= 3


def test_noisy_signal_processed_gracefully(synthetic_leak_signal, trained_model):
    """TC10: Signal with added extreme noise must still be filtered and classified without crashing."""
    sig, sr = synthetic_leak_signal
    # Add strong noise
    heavy_noise = np.random.RandomState(99).randn(len(sig)) * 0.8
    noisy_sig = sig + heavy_noise

    # Pipeline should process smoothly
    filtered = bandpass_filter(noisy_sig, sr=sr)
    norm = normalize(filtered)
    segs = segment(norm, sr=sr)
    assert len(segs) == 5

    df_feat = extract_features_from_signal(noisy_sig, sr=sr)
    X = df_feat[FEATURE_NAMES].values
    preds = trained_model.predict(X)
    assert len(preds) == 5
    assert np.all(np.isin(preds, [0, 1]))


def test_database_save_and_retrieve():
    """TC11: Prediction records must be safely saved to and retrieved from SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        temp_db_path = tmp_file.name

    try:
        init_db(temp_db_path)
        inserted_id = save_result(
            filename="test_pipe_sensor_01.wav",
            prediction="POSSIBLE WATER LEAK DETECTED",
            confidence=95.4,
            db_path=temp_db_path,
        )
        assert inserted_id >= 1

        history_df = get_history(db_path=temp_db_path, limit=10)
        assert len(history_df) == 1
        assert history_df.iloc[0]["filename"] == "test_pipe_sensor_01.wav"
        assert history_df.iloc[0]["prediction"] == "POSSIBLE WATER LEAK DETECTED"
        assert float(history_df.iloc[0]["confidence"]) == pytest.approx(95.4, rel=1e-2)

        # Clear history test
        clear_history(db_path=temp_db_path)
        history_cleared = get_history(db_path=temp_db_path)
        assert len(history_cleared) == 0
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
