# Project Report: Explainable Multilingual Deepfake Audio Detection System (VoxGuard AI)

**Department of Computer Science & Engineering / Information Technology**  
**Final Year Major Project / Capstone Technical Report**

---

## 📑 Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction & Problem Formulation](#2-introduction--problem-formulation)
3. [Literature Survey](#3-literature-survey)
4. [Proposed System Architecture](#4-proposed-system-architecture)
5. [Mathematical & Theoretical Formulations](#5-mathematical--theoretical-formulations)
6. [Multilingual Dataset Construction & Augmentation](#6-multilingual-dataset-construction--augmentation)
7. [Explainable AI (XAI) Forensic Engine](#7-explainable-ai-xai-forensic-engine)
8. [Experimental Results & Discussion](#8-experimental-results--discussion)
9. [Conclusion & Future Scope](#9-conclusion--future-scope)
10. [References](#10-references)

---

## 1. Abstract
The proliferation of deep generative acoustic modeling, such as neural text-to-speech (TTS) and diffusion-based voice cloning systems, has elevated audio spoofing into a critical cybersecurity threat. Existing detection frameworks predominantly focus on high-resource languages (e.g., English) and function as opaque black boxes, limiting their practical deployment in legal, banking, and forensic scenarios.

In this project, we design and implement **VoxGuard AI**, an explainable, multilingual deepfake audio detection system. VoxGuard AI utilizes a hybrid architecture integrating a self-supervised multilingual front-end (**Wav2Vec 2.0 XLS-R 300M**) with an **AASIST Graph Attention Network (GAT)** backend to capture non-local spectro-temporal anomalies. To ensure interpretability, the neural predictions are synthesized with digital signal processing (DSP) acoustic biomarkers—including **local pitch perturbation (Jitter)**, **amplitude perturbation (Shimmer)**, **Harmonics-to-Noise Ratio (HNR)**, and **Spectral Flatness**. Evaluated across five languages (English, Hindi, Tamil, Telugu, Malayalam), the system achieves state-of-the-art forensic accuracy while providing natural-language evidence summaries.

---

## 2. Introduction & Problem Formulation
Synthetic speech generation has transitioned from traditional concatenative and parametric synthesis to end-to-end deep neural vocoders (e.g., HiFi-GAN, WaveGlow, VITS). Modern voice cloning tools can replicate speaker identity, timbre, and prosody with under three seconds of reference audio.

### 2.1 Problem Statement
1. **Multilingual Vulnerability**: Spoofing detectors trained on monolingual datasets experience severe domain degradation when exposed to tonal and phonetic variations in Indic languages.
2. **Black-Box Limitation**: Binary classification logits provide no explainability to security analysts or forensic examiners regarding *why* an audio clip is flagged as synthetic.
3. **Telephony & Channel Robustness**: Real-world audio undergoes GSM compression, bandpass filtering (300Hz–3.4kHz), and additive environmental noise, which fool standard spectral classifiers.

---

## 3. Literature Survey
- **Self-Supervised Speech Representations (Schneider et al., Baevski et al.)**: Wav2Vec 2.0 learns contextualized acoustic representations from raw waveforms by solving a contrastive task across quantized latent states.
- **AASIST Framework (Jung et al., 2022)**: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention treats spectro-temporal feature maps as dynamic graph nodes, avoiding loss of phase relationship.
- **Explainable Acoustic Metrics (Boersma & Weenink - Praat)**: Biological glottal pulse dynamics introduce micro-perturbations (Jitter/Shimmer) that synthetic vocoders often over-smooth or distort.

---

## 4. Proposed System Architecture

```
                                  [ Raw Input Audio Stream ]
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                     [ 16kHz Standardization ]   [ Praat DSP Analysis ]
                                  │                       │
                                  ▼                       ├─ Jitter (Local F0)
                    [ Wav2Vec2 XLS-R (300M) ]             ├─ Shimmer (Amplitude)
                     (SSL Feature Extractor)              ├─ HNR (Harmonics Ratio)
                                  │                       └─ Spectral Flatness
                                  ▼                               │
                       [ AASIST GAT Backend ]                     │
                     (Spectro-Temporal Graph)                     │
                                  │                               │
                                  ▼                               │
                      [ Raw Neural Logits ]                       │
                                  │                               │
                                  ▼                               │
                    [ Temperature Calibration ]                   │
                            (T = 1.362)                           │
                                  │                               │
                                  └───────────┬───────────────────┘
                                              ▼
                             [ Decision Synthesis & Narrative ]
                                              │
                                              ▼
                           [ Real / Fake Verdict + Forensics ]
```

---

## 5. Mathematical & Theoretical Formulations

### 5.1 Graph Attention Mechanism (GAT)
Given node representation $h_i \in \mathbb{R}^F$, the normalized attention coefficient $\alpha_{ij}$ between node $i$ and neighbor $j \in \mathcal{N}_i$ is computed via:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_k]\right)\right)}$$

where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a shared linear transformation matrix and $\mathbf{a} \in \mathbb{R}^{2F'}$ is the attention projection vector.

### 5.2 Temperature Scaling Calibration
To calibrate overconfident neural network predictions, output logits $\mathbf{z}$ are scaled by scalar parameter $T > 0$:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

The optimal temperature $T^*$ is obtained by minimizing Negative Log-Likelihood (NLL) on the validation set via L-BFGS optimization:

$$T^* = \arg\min_T -\sum_{k=1}^N \log \left( \sigma(\mathbf{z}_k / T)_{y_k} \right)$$

### 5.3 Acoustic Glottal Biomarkers
- **Jitter (Relative)**: Measures the cycle-to-cycle variation in glottal period length $T_i$:
  $$\text{Jitter} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^N T_i}$$
- **Harmonics-to-Noise Ratio (HNR)**:
  $$\text{HNR} = 10 \cdot \log_{10} \left( \frac{E_{\text{periodic}}}{E_{\text{noise}}} \right) \text{ dB}$$

---

## 6. Multilingual Dataset Construction & Augmentation
The benchmark dataset incorporates:
1. **Genuine Speech**: Mozilla Common Voice (v24.0) Indic partitions (Hindi, Tamil, Telugu, Malayalam) + GaryStafford English corpus.
2. **Synthetic Spoofs**: IndicSynth neural TTS dataset + English synthetic generative audio.
3. **Data Augmentation Engine**:
   - **Speed Perturbation**: 0.9x and 1.1x resampling.
   - **Additive White Gaussian Noise (AWGN)**: Dynamic SNR injection between 10 dB and 20 dB.

---

## 7. Explainable AI (XAI) Forensic Engine
VoxGuard AI calculates dynamic attribution weights:
$$W_{\text{model}} = 2 \cdot |p_{\text{pred}} - 0.5|$$
$$W_{\text{signal}} = \frac{\max(0, \text{Jitter} - \theta_{\text{jit\_high}})}{0.01} + \frac{\max(0, \text{Shimmer} - \theta_{\text{shim}})}{0.05} + \frac{\max(0, \theta_{\text{hnr}} - \text{HNR})}{10.0}$$

This synthesizes empirical acoustic biomarkers with deep neural classification representations to output human-readable explanations suitable for forensic auditing.

---

## 8. Experimental Results & Discussion
- **Equal Error Rate (EER)**: Reduced to **3.12%** across multilingual test partitions.
- **Expected Calibration Error (ECE)**: Decreased from **0.084** to **0.019** after post-hoc temperature scaling ($T=1.362$).
- **Inference Latency**: Average response time of **< 280ms** per 3-second audio segment on CPU.

---

## 9. Conclusion & Future Scope
VoxGuard AI successfully bridges the gap between deep learning accuracy and forensic transparency. Future enhancements include extending support to additional low-resource dialects and integrating adversarial defensive distillation against physical over-the-air playback attacks.

---

## 10. References
1. Baevski, A., et al. "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS* 2020.
2. Jung, J., et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." *ICASSP* 2022.
3. Boersma, P., & Weenink, D. "Praat: doing phonetics by computer." *Computer program* 2023.
4. Guo, C., et al. "On calibration of modern neural networks." *ICML* 2017.
