# Structural Reliability Analysis
### First-Order Methods, Copula Modelling, and Variance-Reduction Simulation

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/JupyterLab-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/LaTeX-Technical%20Report-008080?style=flat-square&logo=latex&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Structural%20Reliability-2C3E50?style=flat-square" />
  <img src="https://img.shields.io/badge/Methods-FORM%20%7C%20HLRF%20%7C%20MCS%20%7C%20IS-27AE60?style=flat-square" />
</p>

---

## Overview

This project implements a complete computational pipeline for **structural reliability analysis** — the discipline concerned with estimating the probability that an engineering system fails to meet its performance requirements when its parameters are uncertain.

Two structurally distinct reliability problems are solved end-to-end: one centred on a nonlinear limit-state function under contrasting probabilistic models; one data-driven, involving the lateral reliability of a monitored multi-storey building. Both problems are treated with the same sequence of methods — analytical first-order approximation followed by progressively sophisticated simulation — enabling a controlled comparison of accuracy and computational efficiency across techniques.

The implementation covers:

- **Isoprobabilistic transformation** to the independent standard normal space, via Cholesky decomposition for Gaussian vectors and the Nataf (Gaussian copula) construction for non-Gaussian marginals
- **Hasofer–Lind–Rackwitz–Fiessler (HLRF) algorithm** for locating the most probable failure point iteratively in standard normal space
- **First-Order Reliability Method (FORM)** for closed-form failure probability approximation
- **Crude Monte Carlo simulation** as an unbiased baseline estimator
- **Importance sampling** with the original distribution shifted to the design point
- **Importance sampling with a normal proposal** whose covariance is optimised by minimising the IS second moment on a pilot sample
- **Adaptive importance sampling** with iterative proposal updating from weighted failure samples

All methods report point estimates, coefficients of variation, generalised reliability indices $\hat{\beta} = -\Phi^{-1}(\hat{P}_f)$, and 95% confidence intervals.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Repository Structure](#2-repository-structure)
3. [Methods and Theory](#3-methods-and-theory)
4. [Study 1 — Nonlinear Limit-State Function](#4-study-1--nonlinear-limit-state-function)
5. [Study 2 — Data-Driven Building Reliability](#5-study-2--data-driven-building-reliability)
6. [Key Numerical Results](#6-key-numerical-results)
7. [Figures](#7-figures)
8. [Dependencies and Installation](#8-dependencies-and-installation)
9. [Running the Code](#9-running-the-code)
10. [Report](#10-report)
11. [References](#11-references)

---

## 1. Background and Motivation

Structural reliability quantifies the likelihood that a system remains within acceptable performance bounds when its governing parameters — loads, material properties, geometric dimensions — are uncertain. The fundamental quantity of interest is the **probability of failure**:

$$P_f = \Pr[g(\mathbf{X}) \leq 0] = \int_{\{g(\mathbf{x}) \leq 0\}} f_\mathbf{X}(\mathbf{x})\,\mathrm{d}\mathbf{x}$$

where $g(\mathbf{x})$ is the **limit-state function** (positive in the safe domain, non-positive at failure) and $f_\mathbf{X}$ is the joint probability density of the uncertain input vector $\mathbf{X}$.

Direct numerical integration of this multidimensional integral is computationally infeasible in all but the simplest cases. Two families of methods have emerged to address this:

- **First-order analytical methods (FORM)** linearise the failure boundary at its closest point to the origin in standard normal space, enabling a closed-form approximation via the standard normal CDF.
- **Simulation-based methods** use random sampling combined with variance-reduction techniques to estimate $P_f$ efficiently, even for rare events where crude Monte Carlo requires prohibitively many samples.

This project demonstrates both families on problems of increasing complexity: from a two-dimensional analytical case to a four-variable, data-driven structural assessment where distributions are fitted from real monitoring records.

---

## 2. Repository Structure

```
.
├── Data/
│   ├── Data_K.csv            # Monitored storey stiffness K₁, K₂ [N/mm] — ~1,000,000 rows
│   └── Data_P.csv            # Measured horizontal loads P₁, P₂ [N]    — ~1,000,000 rows
│
├── notebook/
│   ├── problem_1.ipynb       # Study 1: analytical nonlinear limit-state function
│   └── problem_2.ipynb       # Study 2: shear building, data-driven reliability
│
└── report/
    └──  HW6_Report.pdf        # Compiled technical report
```

Each notebook is self-contained with Markdown cells providing the complete mathematical derivation for every implemented method. The notebooks are designed to be read as technical documents, not only as executable scripts.

The large CSV datasets are subsampled reproducibly at 10,000 rows (NumPy seed = 42) for distribution fitting. All statistical results are stable with respect to the full million-row dataset.

---

## 3. Methods and Theory

### 3.1 Standard Normal Space and Isoprobabilistic Transformation

All reliability algorithms operate in the **independent standard normal space** $\mathbb{U}$, where the random vector $\mathbf{U}$ has uncorrelated $\mathcal{N}(0,1)$ components. An isoprobabilistic transformation $T: \mathbf{x} \mapsto \mathbf{u}$ converts the original failure probability integral into:

$$P_f = \int_{\{G(\mathbf{u}) \leq 0\}} \varphi(\mathbf{u})\,\mathrm{d}\mathbf{u}, \qquad G(\mathbf{u}) = g\!\left(T^{-1}(\mathbf{u})\right)$$

where $\varphi(\mathbf{u}) = (2\pi)^{-n/2}\exp(-\tfrac{1}{2}\|\mathbf{u}\|^2)$ is the standard multivariate normal density. Because this density is radially symmetric, the point on the failure surface $G(\mathbf{u}) = 0$ closest to the origin is simultaneously the point of highest probability density — the **design point** or most probable failure point.

Two transformation types are implemented depending on the marginal distributions:

**Cholesky transformation (jointly Gaussian)**

For a correlated Gaussian vector with mean $\boldsymbol{\mu}$, standard deviations $\mathbf{D} = \text{diag}(\sigma_i)$, and correlation matrix $\mathbf{R} = \mathbf{L}\mathbf{L}^\top$:

$$\mathbf{U} = \mathbf{L}^{-1}\mathbf{D}^{-1}(\mathbf{X} - \boldsymbol{\mu}), \qquad \mathbf{X} = \boldsymbol{\mu} + \mathbf{D}\mathbf{L}\mathbf{U}$$

**Nataf transformation (non-Gaussian marginals, Gaussian copula)**

Each marginal is first mapped to a standard normal through its own CDF:

$$\tilde{U}_i = \Phi^{-1}\!\left(F_{X_i}(x_i)\right)$$

The correlated normal vector $\tilde{\mathbf{U}}$ is then decorrelated using the Cholesky factor of the copula correlation matrix $\mathbf{R}_0 = \mathbf{L}_0\mathbf{L}_0^\top$:

$$\mathbf{U} = \mathbf{L}_0^{-1}\tilde{\mathbf{U}}$$

The copula correlation $\rho_0$ differs from the target Pearson correlation $\rho$ by a correction factor $C$ that accounts for distortion introduced by the non-Gaussian marginal transforms (Nataf approximation).

---

### 3.2 HLRF Algorithm

The design point $\mathbf{u}^*$ solves the constrained optimisation problem:

$$\mathbf{u}^* = \underset{\mathbf{u}}{\arg\min}\;\|\mathbf{u}\| \quad \text{subject to} \quad G(\mathbf{u}) = 0$$

The Hasofer–Lind–Rackwitz–Fiessler algorithm linearises the constraint at the current iterate and solves the resulting linear problem analytically:

$$\mathbf{u}^{(k+1)} = \frac{\nabla G(\mathbf{u}^{(k)})^\top \mathbf{u}^{(k)} - G(\mathbf{u}^{(k)})}{\|\nabla G(\mathbf{u}^{(k)})\|^2}\,\nabla G(\mathbf{u}^{(k)})$$

The gradient in $\mathbf{U}$-space is propagated via the chain rule through the Jacobian $\mathbf{J}_{XU} = \mathrm{d}\mathbf{x}/\mathrm{d}\mathbf{u}$:

$$\nabla_\mathbf{u} G = \mathbf{J}_{XU}^\top \nabla_\mathbf{x} g$$

For Gaussian transformations this Jacobian is the constant matrix $\mathbf{D}\mathbf{L}$. For Nataf transformations it involves the ratio of marginal PDF to standard normal PDF at each component. For the four-variable problem in Study 2, gradients are computed by central finite differences.

---

### 3.3 First-Order Reliability Method (FORM)

Once the HLRF algorithm converges to $\mathbf{u}^*$, the **reliability index** is:

$$\beta = \|\mathbf{u}^*\|$$

FORM approximates the failure surface by its tangent hyperplane at $\mathbf{u}^*$. For a hyperplane in standard normal space the failure probability is:

$$P_f^{\text{FORM}} = \Phi(-\beta)$$

where $\Phi$ is the standard normal CDF. For nonlinear limit-state functions this is a first-order approximation; the accuracy depends on the curvature of the failure surface at the design point and is verified against the simulation-based estimates.

---
### 3.4 Simulation-Based Estimation

#### Crude Monte Carlo (CMC)

The unbiased direct estimator is:

```math
\hat{P}_f^{\mathrm{CMC}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[g(\mathbf{X}^{(i)}) \leq 0\right]
```

The approximate coefficient of variation is:

```math
\mathrm{CoV}
\approx
\frac{1}{\sqrt{N P_f}}
```

For rare events, such as (P_f \sim 10^{-5}), achieving (\mathrm{CoV} \leq 10%) requires approximately (N \geq 10^7) samples. This makes crude Monte Carlo simulation impractical without variance reduction.

---

#### Importance Sampling — Shifted Proposal

Samples are drawn from a proposal density (h(\mathbf{x})) and weighted by the likelihood ratio:

```math
w(\mathbf{x})
=
\frac{f_{\mathbf{X}}(\mathbf{x})}{h(\mathbf{x})}
```

The importance sampling estimator is:

```math
\hat{P}_f^{\mathrm{IS}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\!\left[g(\mathbf{X}^{(i)}) \leq 0\right]
w(\mathbf{X}^{(i)})
```

The proposal is the original distribution shifted to be centred at the design point (\mathbf{u}^*) in standard normal space. For a Gaussian shift, the log-weight is:

```math
\ln w(\mathbf{u})
=
(\mathbf{u}^*)^\top \mathbf{u}
-
\frac{1}{2}
\|\mathbf{u}^*\|^2
```

---

#### Importance Sampling with Optimised Normal Proposal

A normal proposal is used in standard normal space:

```math
h(\mathbf{u})
=
\mathcal{N}(\mathbf{u}^*,\boldsymbol{\Sigma}_h)
```

where (\boldsymbol{\Sigma}_h) is chosen to minimise the importance sampling second moment on a pilot sample:

```math
\mathbb{E}_h
\left[
\mathbf{1}^2 w^2
\right]
```

The log-weight is:

```math
\ln w(\mathbf{u})
=
-\frac{1}{2}
\|\mathbf{u}\|^2
+
\frac{1}{2}
(\mathbf{u}-\mathbf{u}^*)^\top
\boldsymbol{\Sigma}_h^{-1}
(\mathbf{u}-\mathbf{u}^*)
+
\frac{1}{2}
\ln
\left|
\boldsymbol{\Sigma}_h
\right|
```

---

#### Adaptive Importance Sampling (AIS)

Starting from the distribution mean, the proposal distribution is updated over successive rounds using the failure-weighted mean and covariance.

The updated proposal mean is:

```math
\boldsymbol{\mu}_h^{(r+1)}
=
\frac{
\sum_i
w_i
\mathbf{1}_i
\mathbf{u}^{(i)}
}{
\sum_i
w_i
\mathbf{1}_i
}
```

The updated proposal covariance is:

```math
\boldsymbol{\Sigma}_h^{(r+1)}
=
\frac{
\sum_i
w_i
\mathbf{1}_i
\left(
\mathbf{u}^{(i)}
-
\boldsymbol{\mu}_h^{(r+1)}
\right)
\left(
\mathbf{u}^{(i)}
-
\boldsymbol{\mu}_h^{(r+1)}
\right)^\top
}{
\sum_i
w_i
\mathbf{1}_i
}
```

---

## 4. Study 1 — Nonlinear Limit-State Function

### Problem Definition

$$g(X_1, X_2) = X_1^2 - 2X_2$$

Failure is defined as $g \leq 0$, equivalently $X_2 \geq X_1^2/2$. The failure boundary is the parabola $X_2 = X_1^2/2$. Both random variables share the first two moments ($\mu_{X_1}=13$, $\sigma_{X_1}=2$, $\mu_{X_2}=18$, $\sigma_{X_2}=5$, $\rho=0.5$) but are modelled under two contrasting distributional assumptions:

| | **Model A** | **Model B** |
|---|---|---|
| $X_1$ distribution | Gaussian | LogNormal |
| $X_2$ distribution | Gaussian | Gumbel Type-I (maximum) |
| Dependence structure | Bivariate Gaussian | Gaussian copula, $\rho_0 = C\rho = 1.023 \times 0.5 = 0.5115$ |

### Model A — Jointly Gaussian

The Cholesky transformation gives $X_1 = 13 + 2U_1$ and $X_2 = 18 + 5(0.5\,U_1 + 0.8660\,U_2)$. The analytical gradient $\nabla_\mathbf{x} g = (2x_1,\,-2)^\top$ is mapped to $\mathbf{U}$-space via $\nabla_\mathbf{u} G = (\mathbf{D}\mathbf{L})^\top \nabla_\mathbf{x} g$. HLRF converges in 17 iterations from the origin.

**FORM result:** $\beta_A = 4.027$, $P_f = 2.83 \times 10^{-5}$, design point $\mathbf{x}^* = (5.744,\; 16.498)$.

The design point is dominated by a large reduction of $X_1$ from its mean of 13 to 5.74. Reducing $X_1$ collapses the resistance term $X_1^2$ substantially, making failure possible without a large movement in $X_2$. The lower tail of $X_1$ is the primary driver of failure under the Gaussian model.

### Model B — LogNormal + Gumbel + Gaussian Copula

LogNormal parameters: $\sigma_{\ln} = \sqrt{\ln(1 + \sigma_{X_1}^2/\mu_{X_1}^2)} = 0.15295$, $\mu_{\ln} = 2.55325$.

Gumbel parameters: $\beta_G = \pi/(\sigma_{X_2}\sqrt{6}) = 0.25651$, $u_G = \mu_{X_2} - \gamma_E/\beta_G = 15.7497$, where $\gamma_E \approx 0.5772$ is the Euler–Mascheroni constant.

Because the Gumbel CDF evaluated at its mean does not equal 0.5, the transformed mean is not at the origin of $\mathbf{U}$-space; the algorithm starts at $(0.077,\, 0.161)$ and converges in 11 iterations.

**FORM result:** $\beta_B = 5.313$, $P_f = 5.39 \times 10^{-8}$, design point $\mathbf{x}^* = (8.677,\; 37.646)$.

Compared to Model A, the design point shifts to a much larger $X_2$ (16.5 → 37.6) with a less severe reduction of $X_1$ (13 → 8.7). The right-skewed Gumbel marginal of $X_2$ places heavier probability on large values, making large-$X_2$ failure scenarios more accessible. **The ratio of failure probabilities across the two models is $P_{f,A}/P_{f,B} \approx 525$ despite identical first and second moments and correlation**, underscoring the critical dependence of rare-event estimates on distributional tail assumptions.

---

## 5. Study 2 — Data-Driven Building Reliability

### Physical Model

A two-storey shear building with three identical columns per storey. Effective lateral storey stiffnesses are $k_i = 3K_i$. Horizontal loads $P_1$ and $P_2$ act at the first and second floors respectively.

Inverting the $2 \times 2$ static stiffness matrix gives the roof displacement:

$$u_{\text{roof}} = \frac{P_1 + P_2}{3K_1} + \frac{P_2}{3K_2}$$

**Limit-state function** (failure threshold: 100 mm):

$$g(K_1, K_2, P_1, P_2) = 100 - \frac{P_1 + P_2}{3K_1} - \frac{P_2}{3K_2}$$

At the sample means, the mean roof displacement is 87.4 mm, giving a safety margin of 12.6 mm.

### Distribution Fitting

Both datasets contain approximately one million monitoring records. A reproducible subsample of 10,000 rows (seed = 42) is used for all fitting, and Kolmogorov–Smirnov tests confirm distributional adequacy.

**Stiffness (K₁, K₂) — Bivariate Normal:**

| | $K_1$ [N/mm] | $K_2$ [N/mm] |
|---|---:|---:|
| Mean $\mu$ | 7,195.3 | 4,116.4 |
| Std. dev. $\sigma$ | 364.9 | 205.6 |
| Pearson correlation $\rho_K$ | 0.305 | — |
| KS $p$-value | 0.776 | 0.922 |

**Loads (P₁, P₂) — Joint LogNormal:**

| | $P_1$ [N] | $P_2$ [N] |
|---|---:|---:|
| Mean $\mu$ | 100,089 | 301,457 |
| Std. dev. $\sigma$ | 29,974 | 90,208 |
| Coefficient of variation | 0.300 | 0.299 |
| Log-mean $\lambda$ | 11.471 | 12.574 |
| Log-std $\zeta$ | 0.2931 | 0.2928 |
| Log-space correlation $\rho_{\ln P}$ (Nataf) | 0.496 | — |
| KS $p$-value | 0.997 | 0.884 |

### Case 1 — Fixed Nominal Loads (2D problem)

Loads fixed at $P_1^{\text{nom}} = 2.04 \times 10^5$ N and $P_2^{\text{nom}} = 6.12 \times 10^5$ N. Only $(K_1, K_2)$ are random. Transformation via the Cholesky factor of $\boldsymbol{\Sigma}_K$; analytical gradients.

**HLRF converges in 7 iterations:** $K_1^* = 6358.8$ N/mm, $K_2^* = 3564.9$ N/mm, $\beta = 3.097$, $P_f = 9.78 \times 10^{-4}$.

Both stiffness components at the design point lie below their means. The sensitivity vector $\boldsymbol{\alpha} = (-0.740,\,-0.672)$ shows nearly equal contribution from both storeys.

### Case 2 — All Four Variables Random (4D problem)

All four variables are random. The stiffness and load vectors are physically independent, so the $\mathbb{R}^4 \to \mathbf{U}$ transformation is block-diagonal — Cholesky for $(K_1, K_2)$ and Nataf for $(\ln P_1, \ln P_2)$. Gradients are computed by central finite differences.

**HLRF converges in 9 iterations:** $K_1^* = 7074$ N/mm, $K_2^* = 4034$ N/mm, $P_1^* = 158{,}281$ N, $P_2^* = 713{,}245$ N, $\beta = 3.128$, $P_f = 8.81 \times 10^{-4}$.

The stiffness design point values are near their means (standard normal components ≈ −0.33), while the load design point values are 55% and 137% above their nominal values. Load variability is the dominant failure driver in Case 2. The reliability index increases slightly from Case 1 to Case 2 because the nominal loads in Case 1 are already substantially higher than the fitted mean loads, making the fixed-load assumption more conservative.

---

## 6. Key Numerical Results

### Study 1 — FORM

| Model | $\mathbf{u}^*$ | $\mathbf{x}^*$ | $\beta$ | $P_f^{\text{FORM}}$ | HLRF iterations |
|-------|---|---|:---:|:---:|:---:|
| A — Gaussian | $(-3.628,\;1.748)$ | $(5.744,\;16.498)$ | **4.027** | $2.83\times10^{-5}$ | 17 |
| B — LN + Gumbel + Copula | $(-2.567,\;4.652)$ | $(8.677,\;37.646)$ | **5.313** | $5.39\times10^{-8}$ | 11 |

### Study 1 — Simulation, Model A

| Method | $N$ | $\hat{P}_f$ | $\hat{\beta}$ | CoV | 95% CI |
|--------|----:|------------|:---:|:---:|--------|
| Crude MCS | 500,000 | $1.60\times10^{-5}$ | 4.159 | 0.354 | $[4.9\times10^{-6},\;2.7\times10^{-5}]$ |
| IS — shifted original | 50,000 | $2.46\times10^{-5}$ | 4.060 | 0.010 | $[2.41\times10^{-5},\;2.51\times10^{-5}]$ |
| IS — optimised normal | 50,000 | $2.41\times10^{-5}$ | 4.064 | 0.010 | $[2.37\times10^{-5},\;2.46\times10^{-5}]$ |
| Adaptive IS | 50,000 | $2.47\times10^{-5}$ | 4.059 | 0.045 | $[2.25\times10^{-5},\;2.69\times10^{-5}]$ |
| **FORM reference** | — | $2.83\times10^{-5}$ | **4.027** | — | — |

### Study 1 — Simulation, Model B

| Method | $N$ | $\hat{P}_f$ | $\hat{\beta}$ | CoV | 95% CI |
|--------|----:|------------|:---:|:---:|--------|
| Crude MCS | 500,000 | **0** (no failures) | — | — | — |
| IS — shifted original | 50,000 | $4.96\times10^{-8}$ | 5.328 | 0.034 | $[4.63\times10^{-8},\;5.29\times10^{-8}]$ |
| IS — optimised normal | 50,000 | $5.41\times10^{-8}$ | 5.312 | 0.011 | $[5.29\times10^{-8},\;5.53\times10^{-8}]$ |
| Adaptive IS | 50,000 | $4.58\times10^{-8}$ | 5.343 | 0.578 | — |
| **FORM reference** | — | $5.39\times10^{-8}$ | **5.313** | — | — |

Crude MCS produces zero failures for Model B ($P_f \sim 5\times10^{-8}$) even at $N = 500{,}000$. The shifted IS estimator achieves CoV = 3.4% with 50,000 samples; the optimised normal proposal reaches CoV = 1.1% — a 32× efficiency improvement over crude MCS per sample.

### Study 2 — FORM

| Case | Random variables | $\beta$ | $P_f^{\text{FORM}}$ | HLRF iterations |
|------|---|:---:|:---:|:---:|
| 1 — Fixed loads | $K_1,\;K_2$ | **3.097** | $9.78\times10^{-4}$ | 7 |
| 2 — All random | $K_1,\;K_2,\;P_1,\;P_2$ | **3.128** | $8.81\times10^{-4}$ | 9 |

### Study 2 — Simulation

| Case | Method | $N$ | $\hat{P}_f$ | $\hat{\beta}$ | CoV |
|------|--------|----:|------------|:---:|:---:|
| 1 | Crude MCS | 500,000 | $1.148\times10^{-3}$ | 3.049 | 0.042 |
| 1 | IS at design point | 100,000 | $1.063\times10^{-3}$ | 3.072 | 0.006 |
| 1 | IS optimised covariance | 100,000 | $1.063\times10^{-3}$ | 3.072 | 0.006 |
| 2 | Crude MCS | 500,000 | $8.840\times10^{-4}$ | 3.127 | 0.048 |
| 2 | IS at design point | 100,000 | $9.162\times10^{-4}$ | 3.116 | 0.006 |
| 2 | IS optimised covariance | 100,000 | $9.250\times10^{-4}$ | 3.113 | 0.006 |

Importance sampling reduces the CoV from ~4–5% at $N=500{,}000$ to below 0.6% at $N=100{,}000$ — approximately a 50× efficiency gain in variance per sample.

---

## 7. Figures

| File | Description |
|------|-------------|
| `problem1_gaussian_hlrf_design_point.png` | HLRF search sequence, Model A: physical space with parabolic failure boundary and iteration path; standard normal space with reliability circle |
| `problem1_lognormal_gumbel_copula_hlrf_design_point.png` | HLRF search sequence, Model B: distorted failure boundary image in $\mathbf{U}$-space from non-Gaussian marginals |
| `plot3_mc_comparison_both_cases.png` | Bar charts of $\hat{P}_f$ and $\hat{\beta}$ for all simulation methods across both models with 95% CI and FORM reference lines |
| `plot4_adaptive_IS_convergence.png` | Per-stage AIS convergence: $\hat{P}_f$ with confidence band and CoV on dual axes across 10 rounds for both models |
| `Fig1_K_distribution_fitting.png` | Stiffness diagnostics: histograms with fitted normals, joint scatter with confidence ellipses, Q–Q plots |
| `Fig2_P_distribution_fitting.png` | Load diagnostics: histograms with fitted lognormals, log-space scatter, lognormal Q–Q plots |
| `Fig3_HLRF_fixed_loads_design_point.png` | Case 1 HLRF: $K_1$–$K_2$ physical space, standard normal space, convergence history (3-panel) |
| `Fig4_HLRF_probabilistic_loads.png` | Case 2 HLRF: 6-panel projection of the 4D search path onto coordinate planes, plus convergence and $\beta$ evolution |
| `Fig5_comparison_all_methods_without_a.png` | Study 2 reliability index $\hat{\beta}$ comparison across FORM, CMC, IS-DP, IS-Opt |
| `Fig6_probability_of_failure_comparison.png` | Study 2 failure probability comparison with 95% CI error bars |

---

## 8. Dependencies and Installation

### Requirements

```
python     >= 3.10
numpy      >= 1.24
scipy      >= 1.10
matplotlib >= 3.7
tabulate   >= 0.9
jupyterlab >= 4.0
```

### Setup

**pip:**
```bash
pip install numpy scipy matplotlib tabulate jupyterlab
```

**conda (recommended):**
```bash
conda create -n reliability python=3.11
conda activate reliability
conda install numpy scipy matplotlib jupyterlab
pip install tabulate
```


---

## 9. Running the Code

**Step 1 — Data files**

Place `Data_K.csv` and `Data_P.csv` in the `Data/` directory. Both files should be two-column, headerless CSV:

```
Data_K.csv  →  col 1: K₁ [N/mm],  col 2: K₂ [N/mm]   (~1,000,000 rows)
Data_P.csv  →  col 1: P₁ [N],     col 2: P₂ [N]        (~1,000,000 rows)
```

**Step 2 — Launch JupyterLab**

```bash
cd notebook/
jupyter lab
```

**Step 3 — Execute notebooks**

`problem_1.ipynb` — standalone, no data files required. Run all cells in sequence.

`problem_2.ipynb` — requires `../Data/Data_K.csv` and `../Data/Data_P.csv`. Run all cells in sequence.

Expected runtimes on a standard laptop:

| Notebook | Approximate runtime |
|----------|:---:|
| `problem_1.ipynb` | ~45 s |
| `problem_2.ipynb` | ~90 s |

The dominant cost in both notebooks is the 500,000-sample crude Monte Carlo run.

---

## 10. Report

The technical report (`report/HW6_Report.pdf`) is 28 pages, typeset in LaTeX with full mathematical derivations, cross-referenced equations, figures, and tables.

| Section | Content |
|---------|---------|
| Abstract | Summary of both studies, principal numerical findings |
| §1 — Theory | Standard normal space, HLRF, FORM, all four simulation estimators with variance formulas |
| §2 — Study 1 | Model A and B transformations, HLRF iteration tables, FORM results, simulation comparison |
| §3 — Study 2 | Physical model, LSF derivation, distribution fitting, Case 1 and 2 HLRF, simulation results |
| §4 — Discussion | Cross-study model sensitivity, FORM accuracy assessment, simulation efficiency analysis |
| Appendix A | Consolidated FORM results for all four analyses |
| Appendix B | Complete fitted distribution parameter table for Study 2 |

---

## 11. References

Hasofer, A. M. & Lind, N. C. (1974). Exact and invariant second-moment code format. *Journal of the Engineering Mechanics Division*, 100(1), 111–121.

Rackwitz, R. & Fiessler, B. (1978). Structural reliability under combined random load sequences. *Computers & Structures*, 9(5), 489–494.

Nataf, A. (1962). Détermination des distributions dont les marges sont données. *Comptes Rendus de l'Académie des Sciences*, 225, 42–43.

Der Kiureghian, A. & Liu, P.-L. (1986). Structural reliability under incomplete probability information. *Journal of Engineering Mechanics*, 112(1), 85–104.

Haldar, A. & Mahadevan, S. (2000). *Probability, Reliability and Statistical Methods in Engineering Design*. John Wiley & Sons.

Melchers, R. E. & Beck, A. T. (2018). *Structural Reliability Analysis and Prediction* (3rd ed.). John Wiley & Sons.

Rubinstein, R. Y. & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method* (3rd ed.). John Wiley & Sons.

---

<p align="center">
  <sub>
    Oluwakorede Hezekiah Oyewole &nbsp;·&nbsp;
    MSc in Mechanics of Sustainable Materials and Structures &nbsp;·&nbsp;
    University of Trento &nbsp;·&nbsp; 2025
  </sub>
</p>
