# AI in Power Electronics — Predictive Maintenance & Prognostics

Application of **Machine Learning and data-driven prognostics** for health monitoring and Remaining Useful Life (RUL) prediction of critical power-electronic components.

This repository combines two Supervised Learning Projects (SLPs) completed at **IIT Bombay**, focusing on predictive maintenance of **Power MOSFETs** and **Capacitors**.

**Guide:** Prof. Alankar  
**Co-Guide:** Prof. Prabhu Ramachandran  
**Indian Institute of Technology Bombay (IIT Bombay)**

---

## Overview

Power-electronic components operate under high electrical and thermal stresses, causing gradual degradation that can eventually lead to system failure.

Instead of relying only on reactive or scheduled maintenance, these projects explore **AI-driven predictive maintenance** by learning degradation patterns from electrical measurements.

The work focuses on two critical components:

### 1. Power MOSFET Prognostics
**State-of-Health (SoH) Assessment & Remaining Useful Life (RUL) Prediction**

### 2. Capacitor Prognostics
**Degradation Analysis, Health Monitoring & ML-Based RUL Prediction**

Together, the projects explore the pipeline:

**Electrical Measurements → Signal Processing → Degradation Features → Health Assessment → Failure Detection → RUL Prediction**

---

# Project 1 — Power MOSFET SoH & RUL Prediction

## Objective

Develop a machine-learning pipeline capable of:

- Detecting degradation in power MOSFETs
- Classifying their current State of Health (SoH)
- Predicting Remaining Useful Life (RUL)
- Supporting condition-based and predictive maintenance

The project focuses on **wear-out failures** caused by long-term thermo-mechanical stress rather than sudden catastrophic failures.

---

## Accelerated Life Testing

Degradation data was obtained from **Accelerated Life Tests (ALT)** performed on approximately 20 MOSFETs.

During repeated power cycles, the following signals were recorded:

- Drain-source voltage — `VDS(t)`
- Drain current — `ID(t)`
- Device temperature
- Cycle index / timestamp

The repeated heating and cooling cycles accelerate ageing mechanisms such as:

- Bond-wire lift-off
- Solder fatigue
- Metallization degradation

---

## Health Indicator — RDS(on)

The primary degradation indicator used was **On-State Resistance — RDS(on)**:

`RDS(on) = VDS / ID`

RDS(on) increases as package-related degradation progresses and can be monitored during device operation.

Two degradation thresholds were used:

- **3% increase in RDS(on)** → Pre-Failure
- **5% increase in RDS(on)** → End of Life (EoL)

---

## Signal Processing & Feature Engineering

Raw switching signals contain measurement noise, overshoot and transient spikes.

The preprocessing pipeline included:

- Signal windowing
- Robust smoothing
- Outlier-resistant averaging
- RDS(on) extraction
- Post-failure data removal
- Sliding-window feature extraction

Statistical features were extracted from:

- VDS
- ID
- RDS(on)

including:

- Mean
- Standard deviation
- Skewness
- Kurtosis

These features were used to represent degradation patterns for the ML models.

---

## Machine Learning Pipeline

A two-stage ML architecture was developed.

### Stage 1 — State-of-Health Classification

A **Random Forest Classifier** predicts:

- `0` → Healthy
- `1` → Pre-Failure

Random Forest was selected for its robustness to noisy and high-dimensional degradation data.

### Stage 2 — Remaining Useful Life Prediction

Once the MOSFET enters the pre-failure state, a **Bayesian Ridge Regressor (BRR)** models the degradation trajectory and predicts the remaining lifetime.

The RUL model uses approximately the last **30 degradation windows** after pre-failure detection.

---

## MOSFET Results

| Task | Model | Performance |
|---|---|---:|
| SoH Classification | Random Forest | **80.7% average accuracy** |
| RUL Prediction | Bayesian Ridge Regression | **1.25% average RMSPE** |
| Best RUL Model | Linear BRR | **0.79% RMSPE** |

The results showed that linear Bayesian Ridge Regression produced more stable RUL estimates than polynomial modelling because MOSFET degradation becomes approximately linear after the pre-failure point.

---

# Project 2 — Capacitor Prognostics & ML-Based RUL Prediction

## Objective

Develop a prognostics framework for capacitor health monitoring using impedance-based degradation indicators and Machine Learning.

The project focuses on:

- Capacitor ageing analysis
- Electrochemical Impedance Spectroscopy (EIS)
- ESR-based degradation tracking
- Health Indicator development
- Failure detection
- Remaining Useful Life prediction

---

## Dataset

The project uses the **NASA Capacitor Electrical Stress Dataset**.

Files analyzed include:

- `ES10.mat`
- `ES12.mat`
- `ES14.mat`

The dataset contains repeated EIS and transient measurements from capacitors subjected to electrical stress.

Important parameters include:

- Frequency
- Real Impedance — Re(Z)
- Imaginary Impedance — Im(Z)
- Impedance Magnitude — |Z|
- Phase
- Voltage
- Current
- Capacitance

---

## ESR-Based Degradation Analysis

**Equivalent Series Resistance (ESR)** was used as the primary resistance-related degradation feature.

At high frequencies, capacitor impedance becomes increasingly dominated by ESR.

ESR was therefore approximated using:

`ESR ≈ median(Re(Z)) at high frequencies`

The processing pipeline included:

- Frequency sorting
- High-frequency impedance extraction
- Outlier removal
- Moving-average smoothing
- Statistical thresholding

---

## Real-World Data Challenges

Analysis of the NASA dataset revealed that the extracted ESR-related feature did **not exhibit perfectly monotonic degradation**.

Observed behaviour included:

- Initial ESR increase
- Early-cycle peaks
- Gradual decrease/stabilization
- Measurement fluctuations
- Non-monotonic degradation patterns

This highlighted an important challenge in real-world prognostics:

> Experimental degradation data is often noisy, non-ideal and affected by measurement and stabilization effects.

Nyquist plot analysis was additionally used to visualize changes in impedance behaviour across ageing cycles.

---

## Health Indicator & Failure Detection

An ESR-based **Health Indicator (HI)** was developed to track relative capacitor health.

A statistical failure threshold was defined as:

`Threshold = Mean(ESR) + 0.8 × Std(ESR)`

The threshold was used to identify:

- Abnormal operating conditions
- Stressed capacitor behaviour
- Potential degradation regions

A basic Remaining Useful Life estimate was then defined as:

`RUL = Failure Cycle − Current Cycle`

---

## Synthetic Degradation Modeling

Because the experimental NASA data exhibited noisy and non-monotonic degradation behaviour, a controlled synthetic degradation dataset was generated.

The synthetic model assumes:

- ESR increases progressively with ageing
- Capacitance decreases with ageing
- Health Indicator decreases with degradation
- Measurement noise represents experimental variability
- Failure occurs when ESR reaches an end-of-life threshold

This provided a controlled environment for validating the complete prognostics pipeline.

---

## ML-Based Capacitor RUL Prediction

A **Random Forest Regressor** was trained using:

- ESR
- Capacitance
- Impedance
- Health Indicator

The model learned the simulated degradation progression and predicted Remaining Useful Life from the extracted health features.

The synthetic experiment demonstrated that the combined degradation indicators could effectively represent health evolution and support ML-based RUL estimation.

---

# Common Prognostics Framework

Although the two projects study different components, both investigate the same broader engineering problem:

> **Can electrical degradation signatures be transformed into useful health indicators and used by Machine Learning models to predict component failure before it occurs?**

The overall methodology can be summarized as:

```text
Electrical / Sensor Measurements
              ↓
       Signal Processing
              ↓
       Feature Engineering
              ↓
    Degradation Indicators
              ↓
     State / Health Assessment
              ↓
       Failure Detection
              ↓
   Remaining Useful Life (RUL)
              ↓
     Predictive Maintenance
```

---

## Technologies & Methods

### Machine Learning

- Random Forest Classification
- Random Forest Regression
- Bayesian Ridge Regression
- Statistical Feature Engineering

### Data & Signal Processing

- Python
- MATLAB
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

### Prognostics

- State-of-Health (SoH) Estimation
- Remaining Useful Life (RUL)
- Health Indicators
- Accelerated Life Testing
- Electrochemical Impedance Spectroscopy
- ESR Analysis
- RDS(on) Analysis
- Nyquist Analysis
- Synthetic Degradation Modeling

---

## Repository Structure

```text
AI-in-Power-Electronics/
│
├── MOSFET_Prognostics/
│   ├── ...
│   └── MOSFET SoH & RUL prediction code
│
├── Capacitor_Prognostics/
│   ├── ...
│   └── Capacitor degradation & RUL prediction code
│
├── datasets/
│   └── ...
│
└── README.md
```

---

## Key Takeaways

- Applied ML to real engineering reliability and predictive-maintenance problems.
- Built pipelines from **raw electrical signals to health and RUL predictions**.
- Explored degradation behaviour across both **power semiconductor and capacitor systems**.
- Worked with noisy experimental data requiring robust preprocessing and feature engineering.
- Compared classification and regression approaches for different prognostics tasks.
- Explored both real-world and synthetic degradation modelling.

---

## Future Work

- Physics-informed Machine Learning
- Deep Learning-based RUL prediction
- Real-time condition monitoring
- Multi-device degradation modelling
- Improved Health Indicator construction
- Uncertainty-aware RUL estimation
- Cross-device generalization
- Deployment for real-time predictive maintenance

---

## Author

**Arpit Jain**  
Indian Institute of Technology Bombay (IIT Bombay)

**Supervised Learning Projects — AI in Power Electronics**
