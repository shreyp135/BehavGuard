# BehaveGuard
> Continuous Behavioral Biometric Authentication via Keystroke Dynamics

BehaveGuard is a real-time behavioral continuous authentication system that verifies users by *how* they type—their motor rhythm, timing patterns, and muscle memory—rather than what they type. By silently capturing key press and release events, BehaveGuard extracts sequential temporal features, matches them against a user's behavioral profile, and flags anomalies in real time.

* [kaggle notebook](https://www.kaggle.com/code/shreyanshakshit/notebook388616783e/notebook)
---

## Key Features

* **Dual-Model Support**: Supports both a baseline **One-Class SVM** (statistical window aggregates) and a deep **LSTM Autoencoder** (sequential temporal dynamics).
* **Key-Specific Normalization**: Computes key-level and digraph-level average timing baselines to capture typing *rhythm shapes* rather than just absolute typing speed.
* **Stable Sequence-to-Sequence Learning**: Resolves vanishing gradient problems in LSTM-AEs by projecting bottleneck latent vectors directly to initialize the decoder's recurrent states $(h_0, c_0)$.
* **Dual-Anomaly Scoring**: Combines reconstruction error (MSE) and bottleneck cluster proximity (Euclidean latent distance) to detect sophisticated impostors.
* **Dynamic Thresholding**: Adjusts the classification threshold locally during a session within a $\pm 20\%$ bound to handle natural user fatigue or stress.
* **Instant Model Conversion**: Includes utilities to train/switch model architectures using stored raw keystrokes, avoiding the need for repeated physical enrollment sessions.

---

## System Architecture & Pipeline

```
  [Keystroke Stream] (Press/Release TS)
          │
          ▼
  [Key-Specific Normalizer] (Uses baseline means for specific keys/digraphs)
          │
          ▼
  [Sequence Windowing] ──► Shape: (50, 7)
          │
          ├─────────────────────────────────────────┐
          ▼ (LSTM Flow)                             ▼ (SVM Flow)
    [Encoder LSTM]                            [Aggregate Stats]
          │ (Compresses sequence)                   │ (Mean, Std, Skew, Kurtosis)
          ▼                                         ▼
    [Latent Space Bottleneck] (16-dim)         [One-Class SVM]
          │                                         │
          ├────────────────────────┐                │
          ▼ (Decoder State Proj)   ▼ (Decoder In)   │
      (h0, c0) state           [50, 16] sequence    │
          │                        │                │
          └───────────┬────────────┘                │
                      ▼                             │
                [Decoder LSTM]                      │
                      │ (Reconstructs sequence)     │
                      ▼                             │
             [Weighted MSE Loss]                    │
                      │                             │
                      ▼                             │
             [Dual Anomaly Score]                   ▼
       (Reconstruction MSE + Latent Dist)   [SVM Anomaly Score]
                      │                             │
                      └────────────┬────────────────┘
                                   ▼
                       [Dynamic Session Scorer]
                                   │
                                   ▼
                       [Real-time Verdict Terminal]
```

---

## Detailed Mechanics

### 1. Key-Specific & Digraph-Specific Normalization
People have different relative speeds for different keys (e.g., holding down the spacebar longer than character keys). Global normalization fails to capture this shape. BehaveGuard records:
* Average dwell times for each specific key.
* Average flight and digraph times for each specific key pair (digraph).

During scoring, each event is normalized against the user's specific baseline for that key/digraph. If an impostor types, their relative rhythm changes (e.g., typing vowels slower and spacebars faster than the genuine user), causing a noticeable signature mismatch.

### 2. LSTM Autoencoder with State Projection
Standard sequence-to-sequence autoencoders suffer from vanishing gradients because the network has to backpropagate error signals through 100 sequential operations (50 steps encoding + 50 steps decoding).

To solve this, BehaveGuard projects the $16$-dimensional latent vector to size $32$ and initializes the decoder LSTM's recurrent initial states $(h_0, c_0)$ with it. This creates a direct connection between the encoder's output and the start of the decoder, keeping backpropagation gradients stable and allowing the network to converge effectively.

### 3. Dual-Anomaly Scoring
To maximize security, the LSTM model evaluates two components:
1. **Reconstruction MSE** (`norm_recon`): Evaluates how well the network can reconstruct the timing sequence.
2. **Latent Distance** (`norm_latent`): Evaluates how close the sequence's compressed vector is to the user's average fingerprint cluster center.

$$\text{Combined Score} = 0.5 \times \text{norm\_recon} + 0.5 \times \text{norm\_latent}$$

---

## Installation

Ensure you have Python 3.12+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/BriskAM/behaviour-auth.git
   cd behaviour-auth
   ```

2. Sync the dependencies (uses [uv](https://github.com/astral-sh/uv) by default):
   ```bash
   uv sync
   ```
   *Alternatively, using standard pip:*
   ```bash
   pip install -r requirements.txt
   pip install torch
   ```

---

## Usage

### 1. Enrollment (Train new models)
Enrollment runs a 5-minute interactive typing session (collecting ~1,100 keystrokes) to build your profile.

```bash
# Enroll using the LSTM model (default)
python main.py enroll --subject alice --model lstm

# Enroll using the SVM model
python main.py enroll --subject alice --model svm
```

### 2. Instant Conversion (Retrain without physical typing)
If you have already completed an enrollment session for a user (e.g. `alice`), you can convert or retrain their profile to a different model instantly using their saved raw keystroke history:

```bash
# Retrain 'alice' to use the LSTM Autoencoder instantly
python scratch/retrain_from_saved.py alice
```

### 3. Live Continuous Scoring
Run live background scoring. The terminal displays real-time session statistics, a live rolling window history, and normalized scores.

```bash
# Start live scoring (automatically detects model type from the profile)
python main.py score --subject alice --duration 300
```

---

## Testing & Verification

Run the integration suite to verify feature extraction shapes, model forward passes, optimization convergence, and profile save/load persistence:
```bash
python scratch/test_lstm.py
```
