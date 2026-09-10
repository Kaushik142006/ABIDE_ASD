# ABIDE ASD Detection Pipeline

<p align="center">
  <b>AI-Based Autism Spectrum Disorder Detection from Resting-State fMRI Functional Connectivity</b>
</p>

<p align="center">
  An end-to-end neuroimaging and machine learning framework for ASD classification using ABIDE rs-fMRI data, CC200 brain parcellation, functional connectivity analysis, statistical feature selection, RBF-SVM classification, and an interactive Streamlit dashboard.
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [What the Project Does](#what-the-project-does)
- [Key Results](#key-results)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [CC200 Atlas](#cc200-atlas)
- [Complete Pipeline](#complete-pipeline)
- [Data Ingestion](#1-data-ingestion)
- [ROI Time-Series](#2-roi-time-series)
- [Signal Preprocessing](#3-signal-preprocessing)
- [Functional Connectivity](#4-functional-connectivity)
- [Fisher Transformation](#5-fisher-transformation)
- [Pivotal Feature Selection](#6-global-pivotal-feature-selection)
- [Feature Scaling](#7-feature-scaling)
- [Production Model](#production-model)
- [Advanced Research Models](#advanced-research-models)
- [DNFN](#1-deep-neuro-fuzzy-network-dnfn)
- [HGSO](#2-henry-gas-solubility-optimization-hgso)
- [FHGO-DNFN](#3-fhgo-dnfn)
- [FHJO](#4-jellyfish-search-optimization-fhjo)
- [FHJO-CNN](#5-fhjo-cnn)
- [Sparse Autoencoder](#6-sparse-autoencoder)
- [FRNN](#7-fuzzy-recurrent-neural-network-frnn)
- [Evaluation](#evaluation)
- [Dashboard](#interactive-streamlit-dashboard)
- [Outputs](#project-outputs)
- [Architecture](#system-architecture)
- [Directory Structure](#project-directory-structure)
- [Technologies](#technologies-used)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Dashboard Usage](#dashboard-usage)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)
- [License](#license)
- [Disclaimer](#disclaimer)

---

# Project Overview

The **ABIDE ASD Detection Pipeline** is an end-to-end artificial intelligence and neuroimaging framework for classifying subjects as either:

- **ASD — Autism Spectrum Disorder**
- **CON — Typical Control**

The project uses resting-state functional magnetic resonance imaging (**rs-fMRI**) data from the **Autism Brain Imaging Data Exchange (ABIDE)** datasets.

Rather than directly applying a classifier to raw neuroimaging data, the project converts the brain signals into a structured representation of functional connectivity between brain regions.

The complete production workflow is:

```text
ABIDE rs-fMRI Data
        |
        v
Subject Metadata and Functional Files
        |
        v
CC200 Brain Parcellation
        |
        v
200 ROI Time-Series
        |
        v
5th-95th Percentile Signal Clipping
        |
        v
ROI Standardization
        |
        v
Pearson Functional Connectivity
        |
        v
Fisher arctanh Transformation
        |
        v
19,900 Unique Connectivity Edges
        |
        v
Welch Two-Sample t-Test Ranking
        |
        v
Top 600 Pivotal Connectivity Edges
        |
        v
StandardScaler
        |
        v
RBF Support Vector Machine
        |
        v
ASD / CON Prediction
        |
        +----------------------+
        |                      |
        v                      v
 Evaluation              Streamlit Dashboard
```

The repository also contains advanced research architectures based on neuro-fuzzy computing and meta-heuristic optimization:

- Deep Neuro-Fuzzy Network (DNFN)
- Henry Gas Solubility Optimization (HGSO)
- FHGO-DNFN
- Jellyfish Search Optimization (FHJO/JSO)
- FHJO-CNN
- EfficientNet-B0 transfer learning
- Sparse Autoencoder (SAE)
- Fuzzy Recurrent Neural Network (FRNN)
- GRU-based sequential modeling

The **RBF-SVM is the primary production classifier associated with the reported benchmark results**.

---

# What the Project Does

The project transforms resting-state fMRI information into machine-learning features and predicts whether a subject belongs to the ASD or typical-control group.

At a high level:

```text
Brain Activity
     |
     v
200 Brain Regions
     |
     v
Brain-Region Time-Series
     |
     v
Functional Connectivity
     |
     v
19,900 Possible Connections
     |
     v
Statistical Feature Selection
     |
     v
600 Pivotal Connections
     |
     v
Machine Learning Classification
     |
     v
ASD / Typical Control
```

The pipeline therefore combines:

1. Neuroimaging preprocessing
2. Brain parcellation
3. Functional connectivity analysis
4. Statistical feature selection
5. Machine learning
6. Deep learning research architectures
7. Meta-heuristic optimization
8. Model evaluation
9. Model serialization
10. Interactive deployment

---

# Key Results

The evaluated production RBF-SVM pipeline achieved:

| Metric | Score |
|---|---:|
| **Accuracy** | **94.8315%** |
| **Balanced Accuracy** | **94.8468%** |
| **ROC-AUC** | **0.9319** |
| **Sensitivity / Recall** | **95.3704%** |
| **Specificity** | **94.3231%** |
| **F1-Score** | **0.9471** |
| **Matthews Correlation Coefficient (MCC)** | **0.8967** |

### Exact Recorded Values

```text
Accuracy                  = 0.9483146067415731
Balanced Accuracy         = 0.9484675723758693
ROC-AUC                   = 0.9319100760148795
Sensitivity               = 0.9537037037037037
Specificity               = 0.9432314410480349
F1-Score                  = 0.9471264367816092
MCC                       = 0.8966632720696869
```

---

# Project Objectives

The primary objectives are:

- Process ABIDE resting-state fMRI data.
- Represent subjects using CC200 ROI time-series.
- Suppress extreme temporal signal values.
- Standardize regional time-series.
- Calculate pairwise functional connectivity.
- Apply Fisher's transformation to connectivity values.
- Generate 19,900 unique connectivity features.
- Statistically rank the connectivity edges.
- Select the top 600 pivotal edges.
- Train a nonlinear RBF-SVM classifier.
- Evaluate classification performance using multiple metrics.
- Serialize the production model and preprocessing artifacts.
- Provide an interactive Streamlit inference dashboard.
- Implement advanced neuro-fuzzy and meta-heuristic research architectures.
- Establish a modular framework for future ASD neuroimaging experiments.

---

# Dataset

## Autism Brain Imaging Data Exchange (ABIDE)

The project is based on the **Autism Brain Imaging Data Exchange (ABIDE)** datasets.

ABIDE is a multisite neuroimaging initiative containing:

- Resting-state fMRI data
- Structural neuroimaging data
- Phenotypic information
- ASD and typical-control subjects
- Data collected across multiple imaging sites

The project structure supports:

```text
ABIDE-I
ABIDE-II
```

The exact experiment should be interpreted according to the specific dataset subset, preprocessing files, and subjects used for a particular run.

---

# CC200 Atlas

The functional data is represented using the **Craddock 200 (CC200)** functional parcellation atlas.

The atlas provides:

```text
N = 200 functional ROIs
```

For a subject, the ROI time-series matrix is represented as:

```text
X in R^(T x 200)
```

where:

- `T` = number of temporal observations
- `200` = number of brain regions

Each column corresponds to the temporal signal of one ROI.

---

# Complete Pipeline

## 1. Data Ingestion

The dataset loader locates functional files and associates them with subject-level metadata.

The repository strictly supports input file formats:

```text
.nii
.jpg
.png
```

The loader can recursively search through nested directories so that the exact folder depth does not have to be hard-coded.

---

# 2. ROI Time-Series

After CC200 parcellation, each subject is represented by the temporal activity of 200 brain regions.

Let:

```text
X = [x1, x2, ..., x200]
```

where:

```text
xj in R^T
```

represents the time-series of ROI `j`.

Thus:

```text
X in R^(T x 200)
```

This representation is the basis for subsequent functional connectivity analysis.

---

# 3. Signal Preprocessing

## 3.1 Percentile-Based Outlier Clipping

Extreme observations can occur because of:

- Scanner noise
- Motion-related artifacts
- Acquisition effects
- Temporal signal abnormalities

Instead of deleting temporal observations, the project clips extreme values.

For every ROI:

```text
P0.05 = 5th percentile
P0.95 = 95th percentile
```

The cleaned value is:

```text
x_clean(t,j) =
min(
    max(x(t,j), P0.05),
    P0.95
)
```

Therefore:

```text
x < P0.05  -> P0.05
x > P0.95  -> P0.95
```

Values inside the percentile range remain unchanged.

---

## 3.2 ROI Standardization

After clipping, every ROI time-series is standardized.

```text
x_hat(t,j) =
(x_clean(t,j) - mu_j)
----------------------
(sigma_j + epsilon)
```

where:

```text
mu_j     = mean of ROI j
sigma_j  = standard deviation of ROI j
epsilon  = 1e-6
```

This produces approximately zero-mean, unit-scale regional signals.

---

# 4. Functional Connectivity

Functional connectivity measures statistical relationships between brain regions.

The project calculates Pearson correlation between every pair of standardized ROI signals.

For regions `i` and `j`:

```text
r_ij =
sum(
    x_hat(t,i) * x_hat(t,j)
)
---------------------------
T
```

The resulting matrix is:

```text
R in R^(200 x 200)
```

Because the matrix is symmetric:

```text
R_ij = R_ji
```

and the diagonal represents self-correlation.

Only unique pairwise connections are retained.

---

# 5. Fisher Transformation

Pearson correlations are bounded between:

```text
-1 <= r <= 1
```

To obtain a representation with more stable statistical properties, Fisher's transformation is applied.

Before transformation:

```text
r = clip(r, -0.9999, 0.9999)
```

Then:

```text
z = arctanh(r)
```

or equivalently:

```text
z =
1/2 * ln(
    (1 + r) / (1 - r)
)
```

This produces the Fisher-transformed functional connectivity representation.

---

# 6. Global Pivotal Feature Selection

A `200 x 200` connectivity matrix contains a large number of pairwise connections.

The number of unique connections is:

```text
M = N(N - 1) / 2

M = 200(199) / 2

M = 19,900
```

Therefore, every subject initially produces:

```text
19,900 connectivity features
```

Using all features can increase computational complexity and the risk of overfitting.

The project therefore performs global statistical feature ranking.

---

## Welch Two-Sample t-Test

For every edge `k`, the ASD and control distributions are compared.

The test statistic is:

```text
t_k =
(mean_ASD - mean_CON)
------------------------------
sqrt(
    variance_ASD / n_ASD
    +
    variance_CON / n_CON
)
```

The absolute value is used for ranking:

```text
|t_k|
```

The 600 edges with the largest absolute t-statistics are retained.

Conceptually:

```text
19,900 Edges
      |
      v
Welch t-Test
      |
      v
Rank by |t|
      |
      v
Top 600
```

---

# 7. Feature Scaling

The selected 600 pivotal connectivity features are standardized before classification.

The project uses:

```text
StandardScaler
```

The scaler is fitted on the training data and saved for reuse during inference.

The final model input is:

```text
X* in R^(n x 600)
```

where:

```text
n = number of subjects
```

---

# Production Model

# RBF Support Vector Machine

The primary production classifier is a:

**Support Vector Machine using an RBF (Radial Basis Function) kernel.**

The model receives:

```text
600 selected connectivity features
```

and predicts:

```text
ASD
```

or:

```text
CON
```

---

## RBF Kernel

The RBF kernel is:

```text
K(x_i, x_j) =
exp(
    -gamma ||x_i - x_j||^2
)
```

The project uses the RBF kernel to model nonlinear relationships in functional connectivity.

---

## Regularization

The production configuration uses:

```text
C = 10.0
```

`C` controls the trade-off between:

- Margin maximization
- Training classification errors

---

## Gamma

The kernel parameter is represented as:

```text
gamma =
1
--------------------
600 * Var(X*)
```

This controls the effective radius of the RBF kernel.

---

## Balanced Class Weights

The classifier uses class-balanced penalties.

The class-specific penalty can be represented as:

```text
C_i =
C * n
--------
2 * n_yi
```

where:

```text
n   = total number of training samples
n_yi = number of samples in class yi
```

This reduces the influence of class-frequency imbalance.

---

# SVM Optimization

The soft-margin SVM can be expressed using the dual formulation:

```text
maximize:

sum(alpha_i)
-
1/2 sum_i sum_j
alpha_i alpha_j y_i y_j K(x_i, x_j)
```

subject to:

```text
0 <= alpha_i <= C_i
```

and:

```text
sum(alpha_i y_i) = 0
```

The resulting nonlinear decision boundary is used for ASD classification.

---

# Advanced Research Models

The repository contains additional research architectures.

These models are included as part of the broader technical framework and are distinct from the production RBF-SVM benchmark.

---

# 1. Deep Neuro-Fuzzy Network (DNFN)

File:

```text
src/dnf_network.py
```

The Deep Neuro-Fuzzy Network combines:

```text
Neural feature extraction
+
Gaussian fuzzy membership functions
+
Fuzzy inference rules
+
Classification
```

The feature vector is mapped into:

```text
D = 64
```

hidden dimensions.

The fuzzy layer contains:

```text
R = 32
```

rules.

---

## Gaussian Membership

For hidden feature `d` and rule `r`:

```text
mu_d,r =
exp(
    -(h_d - c_d,r)^2
    -----------------
    sigma_d,r^2 + epsilon
)
```

where:

```text
h_d       = hidden feature
c_d,r     = fuzzy center
sigma_d,r = fuzzy width
epsilon   = numerical stability term
```

---

## Rule Firing Strength

The project aggregates the fuzzy memberships:

```text
w_r =
sum(
    mu_d,r
)
```

for:

```text
d = 1 ... 64
```

This produces rule-level fuzzy activations.

---

# 2. Henry Gas Solubility Optimization (HGSO)

Files:

```text
src/fhgo.py
src/fhgo_dnfn.py
```

HGSO stands for:

**Henry Gas Solubility Optimization**

The algorithm is inspired by gas solubility and diffusion processes.

Within the DNFN framework, fuzzy centers are treated as optimization variables.

---

## Temperature

The optimization uses an epoch-dependent temperature:

```text
T =
exp(
    -epoch / E_max
)
```

The temperature changes as optimization proceeds.

---

## Solubility

The solubility parameter is updated as:

```text
S_(t+1) =
S_t *
exp(
    -1 / (T + epsilon)
)
```

---

## Fuzzy Center Update

The fuzzy centers are updated through a stochastic search:

```text
c_d,r^(t+1) =
c_d,r^t
+
eta *
N(0,1)
*
S_(t+1)
*
P_d,r
```

where:

```text
eta = 0.01
P_d,r = partial-pressure representation
```

The optimization provides a non-gradient search mechanism for exploring the fuzzy parameter space.

---

# 3. FHGO-DNFN

File:

```text
src/fhgo_dnfn.py
```

This combines:

```text
Deep Neuro-Fuzzy Network
+
Henry Gas Solubility Optimization
```

The conceptual architecture is:

```text
Input Features
      |
      v
Dense Representation
      |
      v
Gaussian Fuzzy Memberships
      |
      v
Fuzzy Rules
      |
      v
Classification
      ^
      |
     HGSO
      |
Fuzzy Parameter Optimization
```

---

# 4. Jellyfish Search Optimization (FHJO)

File:

```text
src/fhjo.py
```

The Jellyfish Search Optimization component is inspired by jellyfish movement in an ocean.

It alternates between:

- Ocean-current exploration
- Swarm-motion exploitation

This provides both global and local search behavior.

---

## Time Control

The optimization uses:

```text
c(t) =
|
(
1 - t / t_max
)
(
2 * rand() - 1
)
|
```

The value determines which optimization behavior is selected.

---

## Ocean Current Exploration

If:

```text
c(t) >= 0.5
```

the attention mask is updated toward the mean/global representation:

```text
M_J^(t+1) =
M_J^t
+
beta *
(
    M_bar_J - M_J^t
)
*
U(0,1)
```

---

## Swarm Exploitation

If:

```text
c(t) < 0.5
```

the attention mask receives a local stochastic update:

```text
M_J^(t+1) =
M_J^t
+
gamma *
N(0, sigma^2)
```

---

# 5. FHJO-CNN

Files:

```text
src/fhjo.py
src/cnn_transfer_learning.py
```

FHJO-CNN combines:

```text
Jellyfish Search Optimization
+
Spatial Attention
+
CNN Transfer Learning
```

A pivotal connectivity representation is organized into a:

```text
50 x 50
```

2D representation for the attention-based CNN branch.

The representation is resized to:

```text
224 x 224
```

before being processed by the CNN backbone.

---

# EfficientNet-B0

The CNN branch uses:

**EfficientNet-B0**

as the transfer-learning backbone.

The conceptual flow is:

```text
Pivotal Connectivity Representation
            |
            v
     Spatial Attention
            ^
            |
           JSO
            |
            v
   Attended Representation
            |
            v
       224 x 224
            |
            v
      EfficientNet-B0
            |
            v
     Feature Representation
            |
            v
       Classification
```

---

# 6. Sparse Autoencoder

The Sparse Autoencoder is used to learn compact latent representations.

Its structure is:

```text
Input
  |
  v
Encoder
  |
  v
Sparse Latent Space
  |
  v
Decoder
  |
  v
Reconstructed Input
```

Sparsity is encouraged using an L1 penalty on the latent representation.

This encourages the network to retain a smaller set of informative latent activations.

---

# 7. Fuzzy Recurrent Neural Network (FRNN)

File:

```text
src/frnn.py
```

The FRNN combines:

```text
Sparse Autoencoder
+
Gaussian Fuzzy Gates
+
GRU
+
Classification
```

The latent representation is structured approximately as:

```text
Z_latent in R^(B x 50 x 32)
```

where:

```text
B  = batch size
50 = sequence/ROI representation
32 = latent feature dimension
```

---

## FRNN Loss

The composite objective is:

```text
L_FRNN =
L_CE(y_hat, y)
+
0.1 ||X - X_hat||_F^2
+
0.01 ||Z_latent||_1
```

The three components represent:

```text
Classification loss
+
Reconstruction loss
+
Sparsity regularization
```

---

# GRU

The FRNN uses a:

```text
2-layer GRU
```

The recurrent architecture is intended to model dependencies across the latent sequential representation.

The complete conceptual workflow is:

```text
Functional Features
       |
       v
Sparse Autoencoder
       |
       v
Latent Representation
       |
       v
Gaussian Fuzzy Gates
       |
       v
2-Layer GRU
       |
       v
Classification
```

---

# Model Summary

| Model | Main Components | Purpose |
|---|---|---|
| **RBF-SVM** | RBF kernel + balanced class weights | **Production classifier** |
| **DNFN** | Dense layers + Gaussian fuzzy rules | Research architecture |
| **HGSO** | Gas-solubility-inspired optimization | Fuzzy parameter optimization |
| **FHGO-DNFN** | HGSO + DNFN | Hybrid research model |
| **FHJO** | Jellyfish Search Optimization | Attention optimization |
| **FHJO-CNN** | FHJO + spatial attention + CNN | Research architecture |
| **EfficientNet-B0** | Transfer learning CNN | CNN backbone |
| **SAE** | Encoder + sparse latent representation | Representation learning |
| **FRNN** | SAE + fuzzy gates + GRU | Sequential research architecture |

---

# Evaluation

The production pipeline was evaluated using an:

```text
80% Training
20% Testing
```

stratified split.

Stratification preserves the relative distribution of ASD and typical-control classes across the split.

---

# Evaluation Metrics

## Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy =
(TP + TN)
----------------
(TP + TN + FP + FN)
```

Result:

```text
94.8315%
```

---

## Balanced Accuracy

Balanced accuracy averages sensitivity and specificity.

```text
Balanced Accuracy =
1/2 *
(
    TP/(TP+FN)
    +
    TN/(TN+FP)
)
```

Result:

```text
94.8468%
```

---

## Sensitivity

Sensitivity measures how many ASD subjects are correctly identified.

```text
Sensitivity =
TP / (TP + FN)
```

Result:

```text
95.3704%
```

---

## Specificity

Specificity measures how many typical-control subjects are correctly identified.

```text
Specificity =
TN / (TN + FP)
```

Result:

```text
94.3231%
```

---

## F1-Score

F1 combines precision and recall.

```text
F1 =
2 * Precision * Recall
----------------------
Precision + Recall
```

Result:

```text
0.9471
```

---

## ROC-AUC

ROC-AUC represents the area under the Receiver Operating Characteristic curve.

Result:

```text
0.9319
```

A higher value indicates stronger class discrimination over different classification thresholds.

---

## Matthews Correlation Coefficient

MCC provides a correlation-based assessment of binary classification.

```text
MCC =
(TP*TN - FP*FN)
-----------------------------------------
sqrt(
    (TP+FP)
    (TP+FN)
    (TN+FP)
    (TN+FN)
)
```

Result:

```text
0.8967
```

---

# Results Summary

```text
+----------------------+------------+
| Metric               | Result     |
+----------------------+------------+
| Accuracy             | 94.8315%   |
| Balanced Accuracy    | 94.8468%   |
| ROC-AUC              | 0.9319     |
| Sensitivity          | 95.3704%   |
| Specificity          | 94.3231%   |
| F1-Score             | 0.9471     |
| MCC                  | 0.8967     |
+----------------------+------------+
```

---

# Confusion Matrix

The evaluation pipeline generates a confusion matrix containing:

```text
                    Predicted
                  ASD       CON

Actual ASD        TP        FN

Actual CON        FP        TN
```

This enables analysis of:

- True ASD detections
- Missed ASD cases
- Correct control predictions
- False ASD predictions

The generated figure can be stored in:

```text
results/figures/confusion_matrix.png
```

---

# Project Outputs

The project produces several categories of outputs.

## 1. Classification Output

The primary prediction is:

```text
ASD
```

or:

```text
Typical Control (CON)
```

---

## 2. Prediction Probability

Where probability estimates are supported by the trained model, the dashboard can display the estimated class probability distribution.

Example:

```text
ASD Probability: 0.91
CON Probability: 0.09
```

The exact values depend on the input subject and trained model.

---

## 3. Serialized Model

The trained production pipeline can be stored as:

```text
results/models/abide_svm_pipeline.pkl
```

The serialized artifact can contain:

```text
RBF-SVM
StandardScaler
Selected feature indices
```

This allows the model to be reused without retraining.

---

## 4. Evaluation Table

The evaluation metrics are stored as:

```text
results/tables/evaluation_metrics.csv
```

Example:

```csv
Accuracy,Balanced Accuracy,ROC-AUC,Sensitivity,Specificity,F1-Score,MCC
0.9483146067415731,0.9484675723758693,0.9319100760148795,0.9537037037037037,0.9432314410480349,0.9471264367816092,0.8966632720696869
```

---

## 5. Evaluation Figures

Example:

```text
results/figures/confusion_matrix.png
```

Additional figures can be generated depending on the evaluation configuration.

---

## 6. Intermediate Features

The pipeline can produce:

```text
ROI time-series
Functional connectivity matrices
Fisher-transformed connectivity matrices
19,900-edge feature vectors
600-edge pivotal feature vectors
Scaled feature matrices
```

These can be stored under:

```text
data/processed/
data/features/
```

---

# Interactive Streamlit Dashboard

The project includes an interactive dashboard implemented in:

```text
frontend.py
```

The dashboard provides a user-facing interface around the trained inference pipeline.

---

# Dashboard Workflow

```text
User
 |
 v
Upload Functional Data
 |
 v
Subject/File Identification
 |
 v
Preprocessing
 |
 v
CC200 Feature Processing
 |
 v
Functional Connectivity
 |
 v
Fisher Transformation
 |
 v
600 Pivotal Features
 |
 v
Saved StandardScaler
 |
 v
Saved RBF-SVM
 |
 v
Prediction
 |
 +------------------+
 |                  |
 v                  v
ASD                CON
 |
 v
Probability / Confidence Display
```

---

# Dashboard Inputs

The project strictly supports inputs in the following formats:

```text
.nii
.jpg
.png
```

### Important

The production SVM does **not** directly classify raw voxel-level NIfTI data.

The classifier expects the feature representation produced by the project's preprocessing pipeline:

```text
NIfTI / Functional Data
        |
        v
Preprocessing
        |
        v
CC200 ROI Time-Series
        |
        v
Functional Connectivity
        |
        v
600 Selected Features
        |
        v
RBF-SVM
```

Therefore, raw NIfTI data must first be transformed into the expected CC200 functional representation.

---

# Dashboard Outputs

The interface can provide:

- Subject identifier
- Predicted class
- ASD/CON probability information
- Confidence visualization
- Prediction summary
- Model inference results

The exact displayed information depends on the implementation of `frontend.py` and the serialized model artifact.

---

# System Architecture

```text
                         ABIDE DATA
                             |
                             v
                  +----------------------+
                  | Dataset Loader       |
                  | dataset_loader.py    |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | CC200 Atlas          |
                  | 200 Functional ROIs  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Preprocessing        |
                  | 5-95% Clipping       |
                  | Standardization      |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Pearson Correlation  |
                  | 200 x 200 Matrix     |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Fisher arctanh       |
                  | Transformation       |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | 19,900 Unique Edges  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Welch t-Test Ranking |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Top 600 Pivotal      |
                  | Connectivity Edges   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | StandardScaler       |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | RBF-SVM              |
                  | C = 10.0              |
                  | Balanced Weights     |
                  +----------+-----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
      +---------------+             +---------------+
      | Evaluation    |             | Streamlit     |
      | Metrics       |             | Dashboard     |
      +---------------+             +---------------+
```

---

# Research Architecture

The advanced models form an additional research branch:

```text
                      600 Selected Features
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
           DNFN             FHJO-CNN          SAE-FRNN
             |                 |                 |
             v                 v                 v
           HGSO          Jellyfish Search       SAE
             |                 |                 |
             |                 v                 v
             |          Spatial Attention     Fuzzy Gates
             |                 |                 |
             |                 v                 v
             |          EfficientNet-B0        GRU
             |                 |                 |
             +-----------------+-----------------+
                               |
                               v
                    Research Model Evaluation
```

---

# Project Directory Structure

```text
ABIDE_ASD_Detection/
│
├── data/
│   ├── raw/
│   │   ├── abide1_data.csv
│   │   ├── abide2_data.csv
│   │   └── ABIDE/
│   │
│   ├── processed/
│   │   └── processed subject data
│   │
│   └── features/
│       ├── connectivity features
│       └── pivotal features
│
├── src/
│   │
│   ├── dataset_loader.py
│   │   ├── Dataset discovery
│   │   ├── Subject matching
│   │   └── Functional file loading
│   │
│   ├── preprocessing.py
│   │   ├── Percentile clipping
│   │   └── Signal preprocessing
│   │
│   ├── roi_extraction.py
│   │   ├── CC200 ROI handling
│   │   └── ROI standardization
│   │
│   ├── functional_connectivity.py
│   │   ├── Pearson correlation
│   │   ├── Fisher transformation
│   │   └── Upper-triangle extraction
│   │
│   ├── pivotal_region.py
│   │   ├── Welch t-tests
│   │   ├── Statistical ranking
│   │   └── Top-600 edge selection
│   │
│   ├── feature_extraction.py
│   │   └── Final feature preparation
│   │
│   ├── dnf_network.py
│   │   └── Deep Neuro-Fuzzy Network
│   │
│   ├── fhgo.py
│   │   └── Henry Gas Solubility Optimization
│   │
│   ├── fhgo_dnfn.py
│   │   └── HGSO + DNFN integration
│   │
│   ├── fhjo.py
│   │   └── Jellyfish Search Optimization
│   │
│   ├── cnn_transfer_learning.py
│   │   └── EfficientNet-B0 integration
│   │
│   ├── frnn.py
│   │   ├── Sparse Autoencoder
│   │   ├── Fuzzy processing
│   │   └── GRU architecture
│   │
│   └── evaluation.py
│       ├── Accuracy
│       ├── Balanced Accuracy
│       ├── Sensitivity
│       ├── Specificity
│       ├── F1
│       ├── ROC-AUC
│       ├── MCC
│       └── Confusion Matrix
│
├── results/
│   ├── figures/
│   │   └── confusion_matrix.png
│   │
│   ├── tables/
│   │   └── evaluation_metrics.csv
│   │
│   └── models/
│       └── abide_svm_pipeline.pkl
│
├── frontend.py
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Technologies Used

## Python

Primary programming language.

---

## NumPy

Used for:

- Numerical computation
- Matrix operations
- Vector manipulation
- Connectivity calculations

---

## Pandas

Used for:

- Metadata processing
- Dataset manipulation
- Evaluation tables
- CSV handling

---

## SciPy

Used for:

- Statistical testing
- Welch two-sample t-tests
- Scientific computations

---

## Scikit-Learn

Used for:

- SVM
- RBF kernel classification
- StandardScaler
- Train/test splitting
- Class weighting
- Evaluation metrics
- Model pipelines

---

## PyTorch

Used for:

- DNFN
- FRNN
- GRU
- Sparse Autoencoder
- Neural architectures
- Research experiments

---

## Torchvision

Used for:

- EfficientNet-B0
- CNN transfer learning
- Image-style connectivity representations

---

## Matplotlib

Used for:

- Evaluation visualizations
- Confusion matrices
- Research figures

---

## Seaborn

Used for:

- Statistical visualization
- Confusion matrix visualization

---

## Joblib

Used for:

- Model serialization
- Pipeline persistence
- Loading trained inference artifacts

---

## Streamlit

Used to build the interactive prediction dashboard.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ABIDE_ASD_Detection.git
cd ABIDE_ASD_Detection
```

Replace the repository URL with the actual GitHub repository.

---

# 2. Create a Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended dependencies:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
joblib
torch
torchvision
streamlit
```

---

# requirements.txt

A recommended requirements file is:

```text
numpy>=1.21
pandas>=1.3
scipy>=1.7
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
joblib>=1.1
torch>=2.0
torchvision>=0.15
streamlit>=1.20
```

---

# Dataset Setup

Place the required dataset files under:

```text
data/raw/
```

Example:

```text
data/
└── raw/
    ├── abide1_data.csv
    ├── abide2_data.csv
    └── ABIDE/
        ├── site_1/
        ├── site_2/
        └── ...
```

For the production functional-connectivity pipeline, the required CC200 ROI time-series files should be available.

Example:

```text
*_rois_cc200.1D
```

Do not commit restricted or copyrighted dataset files to the repository.

---

# Training

Run:

```bash
python main.py
```

The main training workflow is:

```text
Load Metadata
      |
      v
Locate Functional Data
      |
      v
Preprocess ROI Time-Series
      |
      v
Calculate Functional Connectivity
      |
      v
Fisher Transformation
      |
      v
Extract 19,900 Edges
      |
      v
Rank Connectivity Features
      |
      v
Select Top 600
      |
      v
Scale Features
      |
      v
Train RBF-SVM
      |
      v
Evaluate
      |
      v
Save Model
```

---

# Generated Artifacts

After a successful run, the project can generate:

```text
results/
├── models/
│   └── abide_svm_pipeline.pkl
│
├── tables/
│   └── evaluation_metrics.csv
│
└── figures/
    └── confusion_matrix.png
```

---

# Dashboard Usage

Start the Streamlit application:

```bash
python -m streamlit run frontend.py
```

Then open the local Streamlit address shown in the terminal.

The dashboard provides an interactive interface for model inference.

Typical workflow:

```text
Launch Dashboard
      |
      v
Upload Functional Input
      |
      v
Process Subject
      |
      v
Generate Features
      |
      v
Load Saved Pipeline
      |
      v
Generate Prediction
      |
      v
Display ASD / CON Result
```

---

# Expected Production Inference Components

The deployed production pipeline should preserve the same feature-processing sequence used during training:

```text
Input
  |
  v
Preprocessing
  |
  v
CC200 Representation
  |
  v
Functional Connectivity
  |
  v
Fisher Transformation
  |
  v
Selected 600 Features
  |
  v
Training StandardScaler
  |
  v
Training RBF-SVM
  |
  v
Prediction
```

The training-time scaler and selected feature indices must not be replaced by independently fitted inference-time transformations.

---

# Research Significance

Functional connectivity provides a way to study how different brain regions interact during resting-state conditions.

The project focuses on connectivity rather than relying exclusively on individual regional signal amplitudes.

The feature-selection stage attempts to identify connections showing stronger statistical differences between the ASD and control cohorts.

The resulting 600-edge representation provides a substantially smaller feature space than the original 19,900-edge representation.

This makes the classification problem more computationally manageable while retaining connectivity patterns considered most discriminative by the statistical ranking procedure.

---

# Neurobiological Interpretation

The selected connectivity features can potentially be examined in the context of large-scale functional brain networks.

The project is particularly interested in connectivity involving networks such as:

- Default Mode Network (DMN)
- Salience Network (SN)
- Frontoparietal Task Control Network

Connections involving regions such as the:

- Posterior cingulate cortex (PCC)
- Medial prefrontal cortex (mPFC)

can be investigated as part of the interpretation of discriminative connectivity patterns.

However, statistical feature importance should not automatically be interpreted as proof of a causal neurological mechanism.

---

# Reproducibility

To reproduce the reported experiment as closely as possible, the following should remain consistent:

- Dataset version
- Subject inclusion criteria
- Preprocessing files
- CC200 atlas
- Signal preprocessing
- 5th-95th percentile clipping
- Standardization procedure
- Fisher transformation
- Connectivity extraction
- Welch t-test feature ranking
- Number of selected features
- Train/test split
- SVM hyperparameters
- Class weighting
- Software versions
- Random seeds where applicable

The reported metrics should always be associated with the exact experimental configuration that generated them.

---

# Important Scientific Consideration

The reported accuracy comes from a specific evaluation setup.

High performance on a single random or fixed train/test split does not by itself establish clinical validity or cross-site generalization.

For stronger scientific validation, the system should additionally be evaluated using:

```text
Cross-validation
+
Site-wise validation
+
External validation
+
ABIDE-I -> ABIDE-II testing
+
Repeated experiments
```

This is especially important because ABIDE is a multisite dataset and scanner/site effects can influence functional connectivity measurements.

---

# Limitations

The current framework has several limitations.

## 1. Multisite Variability

ABIDE contains data collected using different scanners, acquisition protocols, and imaging sites.

This can introduce site-specific effects.

---

## 2. Generalization

The reported benchmark should not be interpreted as proof that the model will achieve the same performance on an unseen hospital or clinical population.

---

## 3. Feature Selection Validation

Feature selection must be carefully performed within the training data during cross-validation to prevent information leakage.

For a fully leakage-safe evaluation protocol, the t-test ranking and scaler should be fitted only on the training fold.

---

## 4. Clinical Validation

The project is a research prototype.

It has not established clinical diagnostic validity.

---

## 5. Raw NIfTI Processing

Raw NIfTI data requires appropriate preprocessing and CC200 parcellation before the production feature-based SVM can consume it.

A raw NIfTI file should not be treated as equivalent to a ready-to-use CC200 `.1D` representation.

---

## 6. Dataset Bias

ABIDE data may contain demographic, acquisition, and site-related biases.

Such biases can affect machine-learning performance.

---

# Future Work

Potential future development includes:

- Cross-site validation.
- ABIDE-I to ABIDE-II external evaluation.
- Nested cross-validation.
- Site-effect correction.
- ComBat or related harmonization methods.
- Hyperparameter optimization.
- Explainable AI.
- SHAP-based feature interpretation.
- ROI/network-level visualization.
- Graph Neural Networks.
- Dynamic functional connectivity.
- Temporal functional connectivity.
- Multimodal MRI integration.
- Larger external datasets.
- Prospective clinical validation.
- Calibration analysis.
- Confidence-aware prediction.
- Integration of advanced research architectures into the dashboard.

---

# Roadmap

```text
[x] ABIDE data ingestion
[x] CC200 ROI representation
[x] Signal clipping
[x] ROI standardization
[x] Pearson functional connectivity
[x] Fisher transformation
[x] 19,900 connectivity features
[x] Statistical feature ranking
[x] Top 600 feature selection
[x] RBF-SVM production classifier
[x] Evaluation pipeline
[x] Model serialization
[x] Streamlit dashboard
[x] DNFN implementation
[x] HGSO implementation
[x] FHJO implementation
[x] EfficientNet-B0 research branch
[x] SAE-FRNN research branch

[ ] Leakage-safe nested cross-validation
[ ] Cross-site generalization study
[ ] ABIDE-II external validation
[ ] SHAP explainability
[ ] ROI/network visualization
[ ] Model calibration study
[ ] Clinical validation
[ ] Unified research-model comparison dashboard
```

---

# Troubleshooting

## `ValueError: No functional files found recursively`

This generally indicates that the configured functional-data directory is empty or incorrect.

Check that:

```text
data/raw/ABIDE/
```

contains the required functional files.

For the CC200 pipeline, verify that files similar to:

```text
*_rois_cc200.1D
```

exist.

---

## Streamlit Does Not Start

Try:

```bash
python -m streamlit run frontend.py
```

instead of:

```bash
streamlit run frontend.py
```

This ensures Streamlit is executed using the active Python environment.

---

## Model File Not Found

Check that:

```text
results/models/abide_svm_pipeline.pkl
```

exists.

If it does not, run:

```bash
python main.py
```

to generate the trained model artifact according to the configured training pipeline.

---

## Missing Python Packages

Run:

```bash
pip install -r requirements.txt
```

If using a virtual environment, make sure the environment is activated before installation.

---

# Academic Contribution

The project brings together multiple components into one modular ASD neuroimaging framework:

```text
Neuroimaging
      +
Functional Connectivity
      +
Statistical Feature Selection
      +
Kernel Machine Learning
      +
Neuro-Fuzzy Computing
      +
Meta-Heuristic Optimization
      +
Deep Learning
      +
Interactive Deployment
```

The production branch demonstrates the effectiveness of statistical connectivity reduction followed by nonlinear SVM classification, while the additional architectures provide a research platform for investigating alternative optimization and representation-learning strategies.

---

# Citation

If this repository is used in academic work, use the following template and replace the author and repository information with the actual project details:

```bibtex
@software{abide_asd_detection_2026,
  author = {Your Name},
  title = {ABIDE ASD Detection Pipeline},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ABIDE_ASD_Detection}
}
```

---

# Acknowledgments

This project builds upon publicly available neuroimaging research resources, including:

- Autism Brain Imaging Data Exchange (ABIDE)
- Craddock functional parcellation methodology
- Scientific Python ecosystem
- Scikit-learn
- PyTorch
- Torchvision
- Streamlit

The original dataset creators, preprocessing teams, and research communities are acknowledged for making the underlying neuroimaging resources available for scientific research.

---

# License

This repository is intended to be distributed under the MIT License if the repository owner chooses to use that license.

A corresponding `LICENSE` file should be included in the repository.

---

# Disclaimer

This project is intended for:

- Research
- Education
- Machine-learning experimentation
- Neuroimaging research

It is **not a medical device** and should not be used as a standalone system for diagnosing Autism Spectrum Disorder.

The predictions generated by this software must not replace evaluation by qualified healthcare professionals.

The reported machine-learning performance should not be interpreted as evidence of clinical diagnostic validity.

---

# Final Pipeline Summary

The complete production system can be summarized as:

```text
                    ABIDE rs-fMRI
                         |
                         v
                  CC200 Parcellation
                         |
                         v
                    200 ROIs
                         |
                         v
              5-95% Signal Clipping
                         |
                         v
                  ROI Standardization
                         |
                         v
             Pearson Correlation Matrix
                         |
                         v
              Fisher arctanh Transformation
                         |
                         v
                 19,900 Unique Edges
                         |
                         v
                Welch t-Test Ranking
                         |
                         v
                 Top 600 Pivotal Edges
                         |
                         v
                    StandardScaler
                         |
                         v
                    RBF-SVM
                         |
             +-----------+-----------+
             |                       |
             v                       v
       ASD / CON Prediction     Evaluation
             |                       |
             v                       v
      Streamlit Dashboard      Metrics/Figures
```

## Production Benchmark

```text
Accuracy             : 94.8315%
Balanced Accuracy    : 94.8468%
ROC-AUC              : 0.9319
Sensitivity          : 95.3704%
Specificity          : 94.3231%
F1-Score             : 0.9471
MCC                  : 0.8967
```

---

## Project Structure at a Glance

```text
ABIDE_ASD_Detection/
│
├── data/
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── roi_extraction.py
│   ├── functional_connectivity.py
│   ├── pivotal_region.py
│   ├── feature_extraction.py
│   ├── dnf_network.py
│   ├── fhgo.py
│   ├── fhgo_dnfn.py
│   ├── fhjo.py
│   ├── cnn_transfer_learning.py
│   ├── frnn.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── frontend.py
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

<p align="center">
  <b>ABIDE ASD Detection Pipeline</b><br>
  Functional Connectivity + Machine Learning + Neuro-Fuzzy Research + Interactive Deployment
</p>
