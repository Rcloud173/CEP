"""
Streamlit Web Dashboard: Detecting Water Leaks Using Acoustic Signals
Author: Jayprakash Maurya (T.Y. B.Sc. IT - Vidyalankar School of Information Technology)
Project: Detecting Water Leaks Using Acoustic Signals (Community Engagement Project)

Features:
- Home, Upload/Select Audio, Signal Analysis, and Prediction Results.
- Visualizations: Raw vs. Filtered Waveform, FFT Magnitude Spectrum, STFT Spectrogram.
- Machine Learning classification with segment-averaged confidence.
- SQLite database integration for audit logging and history.
- Assistive notice and synthetic data disclaimers.
"""

import os
import sys
import io
import time
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import streamlit as st
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
    FEATURE_NAMES,
)
from application.db import init_db, save_result, get_history, clear_history

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Acoustic Water Leak Detection System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, student-level viva UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .synthetic-banner {
        background: linear-gradient(90deg, #fef3c7, #fde68a);
        color: #92400e;
        padding: 0.65rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        border-left: 5px solid #d97706;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #f3f4f6;
        color: #374151;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #6b7280;
        font-size: 0.88rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-box-normal {
        background-color: #dcfce7;
        color: #14532d;
        padding: 1.4rem;
        border-radius: 10px;
        border: 2px solid #22c55e;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box-leak {
        background-color: #fee2e2;
        color: #7f1d1d;
        padding: 1.4rem;
        border-radius: 10px;
        border: 2px solid #ef4444;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize database
init_db()


# ==========================================
# CACHED MODEL LOADER
# ==========================================
@st.cache_resource
def load_trained_model():
    """Load pre-trained machine learning model artifact."""
    model_path = os.path.join(BASE_PROJECT_DIR, "model", "leak_model.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
        return None


# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/water-pipe.png",
    width=72,
)
st.sidebar.title("Pipeline Acoustic Monitor")
st.sidebar.caption("T.Y. B.Sc. IT | Community Engagement Project")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🏠 Home & Overview", "🔍 Detect & Analyze", "📊 Evaluation Metrics", "📜 Prediction History"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Info**
    - **Student:** Jayprakash S. Maurya
    - **College:** Vidyalankar School of IT (VSIT)
    - **Organization:** Precision Grow
    - **SDG Goal:** SDG 6 (Clean Water)
    """
)

# Always visible top banner



# ==========================================
# VIEW 1: HOME & OVERVIEW
# ==========================================
if nav_choice == "🏠 Home & Overview":
    st.markdown('<div class="main-title">Detecting Water Leaks Using Acoustic Signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An assistive AI signal processing pipeline to detect pipeline leakage before visible surfacing.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            ### Problem Background
            Water distribution pipelines often experience unseen structural leaks due to pressure shocks,
            corrosion, and ground movement. Because pipes are underground or embedded in concrete, small leaks
            can waste millions of liters of treated water before surface dampness is ever observed.

            ### Proposed Acoustic Approach
            When pressurized water escapes from a pipe rupture:
            1. **Turbulent Jet Acoustic Emission:** Generates broadband acoustic vibration (typically 500 Hz to 3000 Hz).
            2. **Digital Preprocessing:** A 4th-order Butterworth bandpass filter eliminates 50 Hz electrical hum and high-frequency noise.
            3. **Feature Extraction:** Computes time-domain (RMS, Zero-Crossing Rate) and frequency-domain characteristics (FFT dominant frequency, spectral centroid, 4 frequency-band energies).
            4. **Machine Learning Classification:** An ensemble classifier categorizes the acoustic pattern into **NORMAL CONDITION** or **POSSIBLE WATER LEAK DETECTED**.
            """
        )

        st.info("💡 **Ready to test?** Navigate to **'🔍 Detect & Analyze'** in the left sidebar to upload or test pre-loaded pipeline audio samples.")

    with col2:
        st.markdown("### Pipeline Architecture")
        st.code(
            """
[Acoustic Sensor / WAV Input]
            │
            ▼
[Validation: Duration, Silence & NaN Check]
            │
            ▼
[Butterworth Bandpass Filter (100 - 3000 Hz)]
            │
            ▼
[Peak Normalization & 1-Second Segmentation]
            │
            ▼
[14 Acoustic Features Extraction]
            │
            ▼
[ML Classifier (Random Forest / Pipeline)]
            │
            ▼
[Confidence Score & Web Dashboard Visualizer]
            │
            ▼
[SQLite Audit Trail Database]
            """,
            language="text",
        )

    st.markdown(
        '<div class="disclaimer-box">ℹ️ <strong>Assistive Tool Disclaimer:</strong> This is an automated indication only. Professional inspection may still be required.</div>',
        unsafe_allow_html=True,
    )


# ==========================================
# VIEW 2: DETECT & ANALYZE
# ==========================================
elif nav_choice == "🔍 Detect & Analyze":
    st.markdown('<div class="main-title">Acoustic Signal Analysis & Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a pipeline audio recording or pick from synthetic test samples.</div>', unsafe_allow_html=True)

    model_artifact = load_trained_model()
    if model_artifact is None:
        st.error("⚠️ Trained model not found. Please run `python model/train.py` first in the project directory.")
        st.stop()

    classifier_pipeline = model_artifact["best_model"]
    model_name = model_artifact.get("best_model_name", "Random Forest")

    # Input method selection
    input_mode = st.radio(
        "Choose Input Source:",
        ["📁 Select Pre-loaded Sample from Dataset", "📤 Upload Custom Audio File (.wav)"],
        horizontal=True,
    )

    audio_bytes = None
    input_filename = ""
    raw_sig = None
    sample_rate = 8000

    dataset_dir = os.path.join(BASE_PROJECT_DIR, "dataset")
    normal_dir = os.path.join(dataset_dir, "normal")
    leak_dir = os.path.join(dataset_dir, "leak")

    if input_mode == "📁 Select Pre-loaded Sample from Dataset":
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            category = st.selectbox("Select Signal Class Category:", ["Normal Pipeline Condition", "Leak Pipeline Condition"])
        with col_c2:
            target_dir = normal_dir if category == "Normal Pipeline Condition" else leak_dir
            if os.path.exists(target_dir):
                available_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(".wav")])
            else:
                available_files = []

            if not available_files:
                st.warning("No audio files found. Please run `python dataset/generate_synthetic.py` first.")
                st.stop()

            selected_file = st.selectbox("Select Sample Audio File:", available_files)
            input_filename = selected_file

        file_full_path = os.path.join(target_dir, selected_file)
        with open(file_full_path, "rb") as f:
            audio_bytes = f.read()

        raw_sig, sample_rate = load_signal(file_full_path, target_sr=8000)

    else:
        uploaded_file = st.file_uploader("Upload a WAV audio file:", type=["wav"])
        if uploaded_file is not None:
            input_filename = uploaded_file.name
            audio_bytes = uploaded_file.read()
            try:
                # Read via io.BytesIO
                data_in, sr_in = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if data_in.ndim > 1:
                    data_in = np.mean(data_in, axis=1)
                if sr_in != 8000:
                    import librosa
                    data_in = librosa.resample(data_in, orig_sr=sr_in, target_sr=8000)
                    sr_in = 8000
                raw_sig = data_in
                sample_rate = sr_in
            except Exception as e:
                st.error(f"❌ Failed to decode audio file '{uploaded_file.name}': {e}. Please ensure it is a valid uncompressed WAV file.")
                st.stop()

    # If audio is available, proceed to validation and analysis
    if raw_sig is not None:
        st.markdown("### Audio Playback")
        st.audio(audio_bytes, format="audio/wav")

        # Step 1: Validation
        is_valid, validation_msg = validate_signal(raw_sig, sr=sample_rate, min_duration=1.0, raise_error=False)

        if not is_valid:
            st.error(f"❌ **Signal Validation Rejected:** {validation_msg}")
            st.info("Requirements: Duration >= 1.0s, non-silent audio, valid numbers without NaN.")
            st.stop()
        else:
            st.success(f"✅ {validation_msg} (Duration: {len(raw_sig)/sample_rate:.2f}s, Sampling Rate: {sample_rate} Hz)")

        # Analysis Trigger Button
        if st.button("🚀 Run Complete Acoustic Detection Pipeline", type="primary"):
            with st.spinner("Processing signal through filtering, feature extraction, and ML classifier..."):
                start_time = time.time()

                # 1. Bandpass filter
                filtered_sig = bandpass_filter(raw_sig, sr=sample_rate, lowcut=100.0, highcut=3000.0)

                # 2. Normalize
                norm_sig = normalize(filtered_sig)

                # 3. Segment into 1-second chunks
                segments = segment(norm_sig, sr=sample_rate, window_sec=1.0, hop_sec=1.0)

                if len(segments) == 0:
                    st.error("Audio is too short to produce at least one full 1-second segment.")
                    st.stop()

                # 4. Feature Extraction per segment
                segment_features = []
                for seg in segments:
                    feat = extract_features_from_segment(seg, sr=sample_rate)
                    segment_features.append(feat)

                feat_df = pd.DataFrame(segment_features)

                # 5. Machine Learning Inference
                X = feat_df[FEATURE_NAMES].values
                segment_probs = classifier_pipeline.predict_proba(X)
                # Leak probability (Class 1) for each segment
                leak_probs = segment_probs[:, 1]
                avg_leak_prob = float(np.mean(leak_probs))

                # Binary decision threshold at 0.50
                is_leak = avg_leak_prob >= 0.50
                if is_leak:
                    prediction_text = "POSSIBLE WATER LEAK DETECTED"
                    confidence_pct = avg_leak_prob * 100.0
                    box_class = "result-box-leak"
                    icon = "🚨"
                else:
                    prediction_text = "NORMAL CONDITION"
                    confidence_pct = (1.0 - avg_leak_prob) * 100.0
                    box_class = "result-box-normal"
                    icon = "✅"

                proc_time_ms = (time.time() - start_time) * 1000

                # 6. Save to SQLite database
                save_result(input_filename, prediction_text, confidence_pct)

            # ==========================================
            # RESULT PRESENTATION
            # ==========================================
            st.markdown("---")
            st.markdown("## 🎯 Detection Result")

            st.markdown(
                f"""
                <div class="{box_class}">
                    <h1 style="margin: 0; font-size: 2.1rem;">{icon} {prediction_text}</h1>
                    <p style="font-size: 1.25rem; margin-top: 0.5rem; font-weight: 500;">
                        Confidence Score: <strong>{confidence_pct:.1f}%</strong>
                        &nbsp;|&nbsp; Model: <strong>{model_name}</strong>
                    </p>
                    <p style="font-size: 0.92rem; margin-bottom: 0; opacity: 0.85;">
                        Processed {len(segments)} segments ({len(raw_sig)/sample_rate:.1f}s audio) in {proc_time_ms:.1f} ms
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Mandatory Disclaimer Box
            st.markdown(
                '<div class="disclaimer-box">⚠️ <strong>Important Note:</strong> This is an automated indication only. Professional inspection may still be required.</div>',
                unsafe_allow_html=True,
            )

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Segments", f"{len(segments)}")
            with m2:
                st.metric("Mean Dominant Freq", f"{feat_df['dominant_freq'].mean():.1f} Hz")
            with m3:
                st.metric("Mean Spectral Centroid", f"{feat_df['spectral_centroid'].mean():.1f} Hz")
            with m4:
                st.metric("Mean Signal RMS", f"{feat_df['rms'].mean():.4f}")

            # ==========================================
            # SIGNAL ANALYSIS VISUALIZATIONS
            # ==========================================
            st.markdown("---")
            st.markdown("## 📈 Signal Processing & Frequency Analysis")

            tab1, tab2, tab3, tab4 = st.tabs([
                "🌊 Waveforms (Raw vs Filtered)",
                "📊 FFT Magnitude Spectrum",
                "🌈 STFT Spectrogram",
                "⏱️ Segment Timeline Breakdown",
            ])

            with tab1:
                st.markdown("#### Comparison: Raw Signal vs. 100-3000 Hz Butterworth Filtered Signal")
                fig1, (ax_raw, ax_filt) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, tight_layout=True)
                t_axis = np.linspace(0, len(raw_sig) / sample_rate, len(raw_sig))

                ax_raw.plot(t_axis, raw_sig, color="#4b5563", lw=0.8)
                ax_raw.set_title("1. Raw Input Acoustic Signal (Before Filtering)", fontsize=11, fontweight="bold")
                ax_raw.set_ylabel("Amplitude")
                ax_raw.grid(True, linestyle="--", alpha=0.5)

                ax_filt.plot(t_axis, norm_sig, color="#2563eb" if not is_leak else "#dc2626", lw=0.8)
                ax_filt.set_title("2. Bandpass Filtered & Peak-Normalized Signal (100 Hz - 3000 Hz)", fontsize=11, fontweight="bold")
                ax_filt.set_xlabel("Time (seconds)")
                ax_filt.set_ylabel("Normalized Amplitude")
                ax_filt.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig1)

            with tab2:
                st.markdown("#### Fast Fourier Transform (FFT) Frequency Spectrum")
                fft_vals = np.fft.rfft(norm_sig)
                fft_freqs = np.fft.rfftfreq(len(norm_sig), 1.0 / sample_rate)
                mag_spectrum = np.abs(fft_vals)

                fig2, ax_fft = plt.subplots(figsize=(11, 4), tight_layout=True)
                ax_fft.plot(fft_freqs, mag_spectrum, color="#7c3aed", lw=1.0)

                # Annotate peak
                dom_idx = np.argmax(mag_spectrum)
                dom_f = fft_freqs[dom_idx]
                ax_fft.axvline(dom_f, color="#ef4444", linestyle="--", alpha=0.8, label=f"Dominant Freq: {dom_f:.1f} Hz")

                # Highlight leak hiss band
                ax_fft.axvspan(500, 3000, color="#fef08a", alpha=0.25, label="Typical Leak Band (500 - 3000 Hz)")

                ax_fft.set_title(f"Acoustic Frequency Spectrum (0 - 4000 Hz) | Dominant Peak at {dom_f:.1f} Hz", fontsize=11, fontweight="bold")
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("FFT Magnitude")
                ax_fft.set_xlim(0, 4000)
                ax_fft.legend(loc="upper right")
                ax_fft.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig2)

            with tab3:
                st.markdown("#### Short-Time Fourier Transform (STFT) Spectrogram")
                f, t_spec, Sxx = signal.spectrogram(norm_sig, fs=sample_rate, nperseg=512, noverlap=256)
                Sxx_db = 10 * np.log10(Sxx + 1e-10)

                fig3, ax_spec = plt.subplots(figsize=(11, 4.5), tight_layout=True)
                mesh = ax_spec.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="plasma")
                ax_spec.set_title("Time-Frequency Intensity Map (Spectrogram)", fontsize=11, fontweight="bold")
                ax_spec.set_xlabel("Time (seconds)")
                ax_spec.set_ylabel("Frequency (Hz)")
                ax_spec.set_ylim(0, 4000)
                cb = fig3.colorbar(mesh, ax=ax_spec, orientation="vertical", pad=0.02)
                cb.set_label("Power Spectral Density (dB)")
                st.pyplot(fig3)

            with tab4:
                st.markdown("#### Segment-by-Segment Leak Probability Timeline")
                seg_df = pd.DataFrame({
                    "Segment": [f"Sec {i+1}" for i in range(len(segments))],
                    "Leak Probability (%)": [round(p * 100.0, 1) for p in leak_probs],
                    "Dominant Freq (Hz)": [round(f, 1) for f in feat_df["dominant_freq"]],
                    "Spectral Centroid (Hz)": [round(c, 1) for c in feat_df["spectral_centroid"]],
                })

                st.dataframe(seg_df, use_container_width=True)

                fig4, ax_bar = plt.subplots(figsize=(10, 3.5), tight_layout=True)
                bar_colors = ["#ef4444" if p >= 0.5 else "#22c55e" for p in leak_probs]
                bars = ax_bar.bar(seg_df["Segment"], seg_df["Leak Probability (%)"], color=bar_colors, width=0.5)
                ax_bar.axhline(50.0, color="#6b7280", linestyle="--", label="Decision Threshold (50%)")
                ax_bar.set_ylim(0, 105)
                ax_bar.set_ylabel("Leak Probability (%)")
                ax_bar.set_title("Per-Second Segment Confidence Progression", fontsize=11, fontweight="bold")
                ax_bar.legend()
                ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
                st.pyplot(fig4)


# ==========================================
# VIEW 3: EVALUATION METRICS
# ==========================================
elif nav_choice == "📊 Evaluation Metrics":
    st.markdown('<div class="main-title">Machine Learning Model Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluated on 18 held-out test audio files (90 unseen 1-second segments).</div>', unsafe_allow_html=True)

    cm_path = os.path.join(BASE_PROJECT_DIR, "model", "confusion_matrix.png")
    comp_path = os.path.join(BASE_PROJECT_DIR, "model", "model_comparison.csv")

    col_m1, col_m2 = st.columns([3, 2])

    with col_m1:
        st.markdown("### Algorithm Performance Comparison")
        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path)
            # Display formatted table
            st.dataframe(
                df_comp.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                }).highlight_max(subset=["F1-Score", "Accuracy"], color="#dcfce7"),
                use_container_width=True,
            )
        else:
            st.info("Model comparison CSV not found. Run `python model/evaluate.py` to generate.")

        st.markdown(
            """
            ### Key ML Design Choices for Viva
            - **Stratified Split by File:** Splitting by audio file (not random segments) ensures segments from the same pipe recording do not leak into train and test sets simultaneously.
            - **StandardScaler Pipeline:** Feature normalization parameters are fitted strictly on training data to avoid test distribution leakage.
            - **Borderline Realistic Scenarios:** Subtle micro-leaks and valve turbulence ensure test accuracy reflects genuine operational overlap rather than artificial 100% perfection.
            """
        )

    with col_m2:
        st.markdown("### Test Set Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Held-out Test Set Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found. Run `python model/evaluate.py` to generate.")


# ==========================================
# VIEW 4: DATABASE HISTORY
# ==========================================
elif nav_choice == "📜 Prediction History":
    st.markdown('<div class="main-title">Acoustic Audit Trail & History</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Logged prediction records stored in local SQLite database.</div>', unsafe_allow_html=True)

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        if st.button("🗑️ Clear History", type="secondary"):
            clear_history()
            st.success("History database cleared.")
            st.rerun()

    history_df = get_history(limit=100)

    if history_df.empty:
        st.info("No prediction records logged yet. Run an analysis in '🔍 Detect & Analyze' to see entries.")
    else:
        st.markdown(f"**Total Records Stored:** {len(history_df)}")

        def highlight_pred(val):
            if "LEAK" in str(val):
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
            return "background-color: #dcfce7; color: #166534; font-weight: bold;"

        # Apply styling safely across pandas versions
        if hasattr(history_df.style, "map"):
            styled_df = history_df.style.map(highlight_pred, subset=["prediction"]).format({
                "confidence": "{:.1f}%"
            })
        else:
            styled_df = history_df.style.applymap(highlight_pred, subset=["prediction"]).format({
                "confidence": "{:.1f}%"
            })
        st.dataframe(styled_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "Detecting Water Leaks Using Acoustic Signals • T.Y. B.Sc. IT Project • Vidyalankar School of Information Technology (VSIT)"
    "</div>",
    unsafe_allow_html=True,
)
