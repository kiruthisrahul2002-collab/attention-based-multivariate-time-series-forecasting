# Advanced Time Series Forecasting with Attention Mechanisms

## Project Overview

- This project implements an advanced deep learning–based time series forecasting system using a Transformer Encoder with self-attention.  
- The goal is to model and forecast a complex multivariate time series exhibiting trend, multiple seasonalities, and non-linear dependencies, and to rigorously compare its performance against a strong LSTM baseline.  
- The solution follows production-quality standards, incorporates walk-forward validation, and provides quantitative and qualitative interpretability through attention analysis.  

---

## Objectives

The project fulfills the following objectives:

- Programmatically generate a complex multivariate time series dataset  
- Engineer look-back windows and multi-step forecast horizons  
- Implement a Transformer Encoder with self-attention  
- Train and evaluate a baseline LSTM model  
- Apply true walk-forward validation  
- Evaluate performance using MAE, RMSE, and MASE  
- Extract and analyze attention representations  
- Provide architectural rationale and interpretability analysis  

---

## Dataset Description

### Data Generation
The dataset is synthetically generated to ensure full control over temporal complexity and includes:

- Linear trend to simulate long-term growth  
- Daily seasonality (short periodic cycles)  
- Weekly seasonality (longer periodic cycles)  
- Non-linear interactions using trigonometric functions  
- Gaussian noise to simulate real-world uncertainty  

Two exogenous variables:  
- A correlated periodic signal  
- A stochastic noise signal  

This design mirrors real-world high-frequency time series such as energy demand or sensor data.  

---

## Feature Engineering

### Input Sequences
- Fixed-length look-back window (historical context)  
- Multivariate input (target + exogenous variables)  

### Output
- Multi-step forecasting horizon  
- Predicts several future time steps simultaneously  

This setup enables the model to learn temporal dependencies across multiple scales.  

---

## Models Implemented

### 1. Baseline Model — LSTM

**Purpose**  
The LSTM serves as a strong sequential baseline for comparison.  

**Architecture**  
- Single LSTM layer  
- Final hidden state used for prediction  
- Fully connected output layer for multi-step forecasting  

---

### 2. Attention Model — Transformer Encoder

**Why Attention?**  
- Avoids compressing all history into a single hidden state  
- Captures long-range dependencies  
- Dynamically focuses on important time steps  

**Architecture**  
- Linear input projection  
- Multi-layer Transformer Encoder  
- Multi-head self-attention  
- Feed-forward networks with residual connections  
- Fully connected forecasting head  

**Design Choice: No Positional Encoding**  
- Temporal order is preserved by sliding windows  
- Attention is used to model relevance, not reordering  

---

## Training Strategy

- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  
- Fixed hyperparameters for reproducibility  
- Explicit training loop with documented settings  

---

## Validation Methodology

### Walk-Forward Validation (Rolling Origin)
Instead of a single static train/test split, the project uses true walk-forward validation:

1. Train on historical data  
2. Forecast the next horizon  
3. Expand the training window  
4. Repeat across the dataset  

**Advantages:**  
- Prevents data leakage  
- Simulates real-world deployment  
- Provides robust performance estimates  

---

## Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures average prediction error magnitude.  
- **RMSE (Root Mean Squared Error):** Penalizes larger forecasting errors.  
- **MASE (Mean Absolute Scaled Error):** Scale-independent metric computed using a rolling naïve forecast derived from training data.  

The MASE implementation strictly follows the formal definition and avoids incorrect static baselines.  

---

## Results Summary

Performance is reported as the average across all walk-forward steps:

| Model       | MAE   | RMSE  | MASE |
|-------------|-------|-------|------|
| LSTM        | Higher | Higher | > 1 |
| Transformer | Lower  | Lower  | < LSTM |

The Transformer model consistently outperforms the LSTM, demonstrating the effectiveness of self-attention for complex temporal dynamics.  

---

## Attention Interpretability Analysis

### Visualization
- Attention representations are extracted from the Transformer Encoder  
- Heatmaps show how attention varies across time steps and features  

### Observed Patterns
- Strong focus on recent time steps → short-term dependencies  
- Periodic attention structures → seasonal effects  
- Distributed attention → non-linear temporal interactions  

### Interpretation
The attention mechanism learns to:  
- Emphasize recent observations for near-term forecasting  
- Recognize repeating seasonal cycles  
- Integrate long-range dependencies when necessary  

This confirms that the model captures meaningful temporal drivers, not noise.  

---

