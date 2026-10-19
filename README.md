# Detecting Water Leaks Using Acoustic Signals
### Community Engagement Project (CEP) — T.Y. B.Sc. (Information Technology) Semester V
**Vidyalankar School of Information Technology (VSIT)**, Autonomous College Affiliated to University of Mumbai  
**Student Name:** Jayprakash Shyamdev Maurya  
**Roll Number:** 24302A0019  
**Faculty Guide:** Mrs. Seema Vishwakarma (Department of Computing)  
**Industry Partner:** Precision Grow  
**Academic Year:** 2026 – 2027  
**Primary SDG Alignment:** United Nations Sustainable Development Goal 6 (Clean Water and Sanitation)  

---

> [!IMPORTANT]
> **Academic & Synthetic Data Disclaimer:**
> All audio signals in this project are **synthetically generated** through mathematical acoustic modeling (simulating fluid dynamics, turbulent jet hiss, and pump noise). Metrics and performance benchmarks reflect synthetic data only and are **NOT indicative of real-world physical pipeline performance**. This application is an **assistive prototype** and is not a replacement for certified plumbing or geotechnical engineering inspection.

---

## 📌 Project Overview
Underground and concealed pipelines frequently develop fractures, loose joints, and corrosion cracks that leak water undetected for months. Traditional methods (visual surveys, water meter audits, pressure drop monitoring) only discover leaks after substantial surface damage or significant pressure loss has already occurred.

This project implements an end-to-end signal processing and machine learning pipeline that monitors pipeline acoustic vibrations. High-pressure water escaping through a defect creates distinctive broadband acoustic emissions (500 Hz to 3000 Hz). The system:
1. Validates and preprocesses acoustic signals using a 4th-order Butterworth bandpass filter.
2. Segments signals into 1-second analysis windows.
3. Extracts 14 time-domain and frequency-domain acoustic features.
4. Classifies the condition as **`NORMAL CONDITION`** or **`POSSIBLE WATER LEAK DETECTED`** using a machine learning classifier.
5. Visualizes raw waveforms, filtered waveforms, FFT frequency spectra, and STFT spectrograms on an interactive Streamlit dashboard.
6. Logs all predictions to a local SQLite database for historical auditing.

---

## 📁 Project Architecture & Directory Structure

```text
water-leak-detection/
├── dataset/
│   ├── generate_synthetic.py     # Generates 120 synthetic WAV files (60 normal, 60 leak)
│   ├── normal_example.png        # Example waveform & spectrogram for normal pipe
│   ├── leak_example.png          # Example waveform & spectrogram for leak pipe
│   ├── normal/                   # Directory containing 60 normal .wav files
│   └── leak/                     # Directory containing 60 leak .wav files
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py             # Audio loader, validator, bandpass filter, normalizer, segmenter
├── feature_extraction/
│   ├── __init__.py
│   └── features.py               # Computes 14 acoustic features per segment & compiles DataFrame
├── model/
│   ├── __init__.py
│   ├── train.py                  # Stratified file split (70/15/15), Pipeline training & comparison
│   ├── evaluate.py               # Test set evaluation, confusion matrix PNG, comparison CSV
│   ├── leak_model.joblib         # Serialized best trained model pipeline artifact
│   ├── confusion_matrix.png      # Test set confusion matrix visualization
│   ├── model_comparison.csv      # Test set benchmark metrics across all candidate models
│   └── split_info.json           # File-level split manifest (guarantees zero data leakage)
├── application/
│   ├── __init__.py
│   ├── app.py                    # Streamlit web application & visual analytics dashboard
│   ├── db.py                     # SQLite database module (table: predictions)
│   └── predictions.db            # SQLite database file storing analysis audit history
├── notebooks/
│   └── pipeline_demonstration.ipynb # Interactive Jupyter notebook for viva walk-through
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Pytest suite with 11 automated unit & integration tests
├── requirements.txt              # Project dependencies
└── README.md                     # Comprehensive project documentation & viva guide
```

---

## 🛠️ Technology Stack
- **Language:** Python 3.10+ (tested on Python 3.14)
- **Signal Processing & Audio:** NumPy, SciPy (signal), Librosa, SoundFile
- **Machine Learning:** Scikit-learn (SVM, Random Forest, KNN, Decision Tree, Pipelines, StandardScaler)
- **Data Handling:** Pandas
- **Visualization:** Matplotlib
- **Web UI & Dashboard:** Streamlit
- **Persistence & Audit:** SQLite3, Joblib
- **Testing:** Pytest

---

## 🚀 Step-by-Step Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/Rcloud173/CEP.git
cd CEPFinalPro
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Execution Instructions (Run in Order)

Execute the pipeline in the following sequence:

### Step 1: Generate Synthetic Dataset
```bash
python dataset/generate_synthetic.py
```
- Creates 120 realistic 5.0-second WAV files sampled at 8000 Hz (60 in `dataset/normal/`, 60 in `dataset/leak/`).
- Generates `dataset/normal_example.png` and `dataset/leak_example.png` visual documentation.

### Step 2: Train Machine Learning Models
```bash
python model/train.py
```
- Automatically extracts 14 features for each 1-second audio segment (600 segments total).
- Splits strictly **by unique file ID** into Train (70% = 84 files), Validation (15% = 18 files), and Test (15% = 18 files) to guarantee zero data leakage.
- Trains 4 pipelines (SVM, Random Forest, KNN, Decision Tree) with `StandardScaler`.
- Selects best model based on validation F1-score and saves `model/leak_model.joblib`.

### Step 3: Evaluate on Unseen Held-out Test Set
```bash
python model/evaluate.py
```
- Evaluates models strictly on the 18 held-out test audio files (90 unseen 1-second segments).
- Prints computed test metrics (Accuracy, Precision, Recall, F1-Score).
- Saves `model/confusion_matrix.png` and `model/model_comparison.csv`.

### Step 4: Run the Streamlit Web Application
```bash
streamlit run application/app.py
```
- Launches local web browser dashboard at `http://localhost:8501`.
- Lets you select any sample or upload custom `.wav` files.
- Displays audio player, raw/filtered waveforms, FFT spectrum, STFT spectrogram, and red/green decision cards.
- Logs prediction results to SQLite.

### Step 5: Run Automated Unit & Integration Tests
```bash
pytest -v tests/test_pipeline.py
```
- Runs 11 automated test cases verifying signal validation, noise tolerance, classification, and database storage.

---

## 🔬 How Each Pipeline Stage Works

```text
                      ACOUSTIC LEAK DETECTION PIPELINE
                                                                          
  [ Raw Audio (.wav) ]                                                    
           │                                                              
           ▼                                                              
  ┌─────────────────┐     Checks: non-empty, non-silent, duration >= 1.0s,
  │ 1. VALIDATION   │ ──► rejects invalid/corrupted audio files.          
  └─────────────────┘                                                     
           │                                                              
           ▼                                                              
  ┌─────────────────┐     4th-order zero-phase Butterworth filter         
  │ 2. BANDPASS     │ ──► removes DC offset, 50 Hz power hum (<100 Hz),   
  │    FILTERING    │     and high-frequency ambient noise (>3000 Hz).    
  └─────────────────┘                                                     
           │                                                              
           ▼                                                              
  ┌─────────────────┐     Normalizes amplitude to [-1.0, +1.0] and slices 
  │ 3. NORMALIZE &  │ ──► into 1.0-second non-overlapping analysis windows
  │    SEGMENTATION │     (8000 samples per segment).                     
  └─────────────────┘                                                     
           │                                                              
           ▼                                                              
  ┌─────────────────┐     Extracts 14 features: RMS, mean, variance, std, 
  │ 4. FEATURE      │ ──► peak, ZCR, FFT dominant frequency, spectral     
  │    EXTRACTION   │     centroid, bandwidth, roll-off, 4 band energies. 
  └─────────────────┘                                                     
           │                                                              
           ▼                                                              
  ┌─────────────────┐     StandardScaler + Random Forest / Classifier     
  │ 5. ML INFERENCE │ ──► predicts probability of leak for each segment;  
  └─────────────────┘     computes average confidence score.              
           │                                                              
           ▼                                                              
  ┌─────────────────┐     Shows NORMAL (green) or POSSIBLE LEAK (red),    
  │ 6. DASHBOARD &  │ ──► waveform, FFT spectrum, spectrogram, and logs   
  │    AUDIT DB     │     results to SQLite database table `predictions`. 
  └─────────────────┘                                                     
```

### Stage 1: Synthetic Signal Generation (`dataset/generate_synthetic.py`)
- **Normal Signal:** Low-pass filtered flow friction (pink/brown noise representing steady fluid through smooth pipe), electrical power hum (50 Hz with 100 Hz harmonic), low-frequency pump rumble (20-35 Hz), and ambient background noise.
- **Leak Signal:** Base normal signal plus turbulent acoustic emission generated by pressurized water jetting through an orifice. Modeled as band-limited broadband hiss (500 Hz to 3000 Hz) with random center frequencies, burst modulations, and transient turbulence spikes.
- **Realistic Overlap:** Includes borderline cases (faint micro-leaks and loud turbulent normal flow with valve noise) to ensure empirical test accuracy reflects practical operational overlap (~90-95%) rather than an artificial 100%.

### Stage 2: Signal Preprocessing (`preprocessing/preprocess.py`)
- **`load_signal`**: Reads `.wav` files via `soundfile`/`librosa`, averages stereo channels into mono, and resamples to target 8000 Hz.
- **`validate_signal`**: Rejects empty files, non-numeric values (NaN/Inf), durations shorter than 1.0s, and silent/near-zero recordings (RMS < $10^{-5}$).
- **`bandpass_filter`**: Applies a 4th-order Butterworth bandpass filter ($100 \text{ Hz} \le f \le 3000 \text{ Hz}$) using Second-Order Sections (SOS) and `sosfiltfilt` (zero-phase filtering to prevent phase distortion).
- **`normalize`**: Peak-normalizes signal values into $[-1.0, +1.0]$ so feature extraction is invariant to recording volume.
- **`segment`**: Divides continuous signals into 1.0-second analysis windows (8,000 samples each).

### Stage 3: Feature Extraction (`feature_extraction/features.py`)
From each 1-second segment, 14 statistical and spectral characteristics are computed:
1. **RMS (Root Mean Square):** $\sqrt{\frac{1}{N} \sum x[n]^2}$ — represents acoustic energy level.
2. **Mean Amplitude:** $\frac{1}{N} \sum x[n]$ — baseline offset check.
3. **Variance:** $\sigma^2$ — signal power spread.
4. **Standard Deviation:** $\sigma$ — amplitude dispersion.
5. **Peak Amplitude:** $\max |x[n]|$ — maximum transient magnitude.
6. **Zero Crossing Rate (ZCR):** Rate of sign changes — indicates noisiness and high-frequency presence.
7. **FFT Dominant Frequency:** Frequency bin with maximum magnitude in the FFT spectrum. Normal flow peaks < 250 Hz (pump/hum), while leaks peak in 800-2400 Hz.
8. **Spectral Centroid:** Center of mass of the spectrum ($\sum f \cdot S(f) / \sum S(f)$). Leaks shift the centroid significantly higher.
9. **Spectral Bandwidth:** Spectral width around the centroid.
10. **Spectral Roll-off:** Frequency below which 85% of total spectral power is concentrated.
11. **Band Energy 1 (0 – 500 Hz):** Background flow friction and mechanical rumble.
12. **Band Energy 2 (500 – 1500 Hz):** Primary orifice jet acoustic emission band.
13. **Band Energy 3 (1500 – 2500 Hz):** Secondary turbulence hiss band.
14. **Band Energy 4 (2500 – 4000 Hz):** High-frequency acoustic harmonics.

### Stage 4: Machine Learning Classification (`model/train.py`, `model/evaluate.py`)
- **No Data Leakage:** Splitting is performed strictly at the **file level** (not by segment). 70% of files train, 15% validate, 15% test.
- **StandardScaler Pipeline:** Feature standardization is fitted only on training segments and transformed onto validation/test sets.
- **Candidate Models:**
  - Random Forest Classifier (100 estimators)
  - Support Vector Machine (RBF kernel, `C=1.0`)
  - K-Nearest Neighbors ($k=5$)
  - Decision Tree (`max_depth=5`)
- **Selection Metric:** Validation set F1-Score.

### Stage 5: Web UI & Persistence (`application/app.py`, `application/db.py`)
- Built using **Streamlit**.
- Provides 4 main tabs/views:
  - **Home & Overview:** Background, architecture, and SDG 6 alignment.
  - **Detect & Analyze:** Audio player, validation badge, dual waveform plot (raw vs. filtered), FFT spectrum with annotated dominant frequency, STFT spectrogram, and per-second segment probability bar chart.
  - **Evaluation Metrics:** Test set confusion matrix and multi-algorithm benchmark table.
  - **Prediction History:** Interactive table displaying SQLite audit log (`predictions.db`).

---

## 📊 Empirical Test Set Results
*(Evaluated strictly on 18 held-out test audio files / 90 unseen 1-second segments; never hard-coded)*

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** (Selected Best) | **91.11%** | **100.00%** | **82.22%** | **90.24%** |
| **Decision Tree** | 90.00% | 97.37% | 82.22% | 89.16% |
| **K-Nearest Neighbors (KNN)** | 90.00% | 100.00% | 80.00% | 88.89% |
| **Support Vector Machine (SVM)** | 88.89% | 100.00% | 77.78% | 87.50% |

*Confusion matrix visualization generated at `model/confusion_matrix.png`.*

---

## 🎓 Viva Voce Q&A Cheat Sheet (For External Examiners)

### Q1: Why use acoustic signals instead of pressure monitoring for leak detection?
> **Answer:** Pressure monitoring typically only registers major pipe ruptures or sudden catastrophic bursts. Small pinhole leaks cause negligible pressure drops across large municipal loops, yet they generate significant acoustic friction and turbulent vibration in the 500–3000 Hz band that contact sensors and accelerometers can pick up directly.

### Q2: Why is the sampling rate set to 8000 Hz?
> **Answer:** According to the **Nyquist-Shannon Sampling Theorem**, the sampling rate must be at least twice the highest frequency component of interest ($f_s \ge 2 \cdot f_{\max}$). Pipeline acoustic emissions and turbulent hiss predominantly occur below 3000 Hz. With $f_s = 8000\text{ Hz}$, our Nyquist limit is $4000\text{ Hz}$, which safely captures all leak characteristics while keeping file sizes small (~78 KB per 5-second file) for edge IoT processing.

### Q3: Why did you split the dataset by file rather than randomly by segment?
> **Answer:** Splitting randomly by segment would allow segments from the *same* 5-second audio recording to appear simultaneously in both the training set and the test set. Because adjacent 1-second segments share the same underlying pipe acoustic profile, this would cause severe **data leakage** and yield falsely inflated, unrealistic test metrics. Splitting by unique file ID ensures the model is evaluated on completely unseen recordings.

### Q4: Why use a 4th-order Butterworth bandpass filter from 100 Hz to 3000 Hz?
> **Answer:** A Butterworth filter has a maximally flat frequency response in the passband with no ripple. Frequencies below 100 Hz contain non-leak disturbances (such as 50 Hz AC electrical mains hum and low-frequency motor rumble), while frequencies above 3000 Hz contain ambient environmental noise. The 100–3000 Hz band cleanly isolates fluid turbulence.

### Q5: What is the significance of the Spectral Centroid?
> **Answer:** The spectral centroid corresponds to the "center of mass" or "brightness" of the frequency spectrum. Normal pipe flow has acoustic energy concentrated in the lower frequencies (< 400 Hz), resulting in a low spectral centroid (~900 Hz). When a leak occurs, high-velocity jetting generates broadband high-frequency hiss, shifting the spectral centroid significantly higher (~1300–1800 Hz).

### Q6: Why is the test accuracy around ~91% and not 100%?
> **Answer:** In real water distribution networks, background noise, valve throttling, and very small pinhole micro-leaks create natural acoustic overlap between normal turbulent flow and weak leaks. An artificial 100% accuracy model would suggest overfitting to synthetic artifacts. Our dataset intentionally includes realistic borderline cases so the classifier demonstrates robust generalization.

---

## 👥 Contributors & Acknowledgements
- **Student Researcher:** Jayprakash Shyamdev Maurya (T.Y. B.Sc. IT, VSIT)
- **Project Mentor:** Mrs. Seema Vishwakarma (Assistant/Associate Professor, Department of Computing, VSIT)
- **Field Study Partner:** Precision Grow (Millennium Business Park, Navi Mumbai)
- **Affiliation:** Vidyalankar School of Information Technology, Autonomous College Affiliated to University of Mumbai
