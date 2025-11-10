# Theoretical Background

This document provides a comprehensive overview of the theoretical foundations underlying the forecasting framework.

## 1. Proper Scoring Rules

### Definition

A scoring rule $S(P, y)$ assigns a numerical score based on the forecast distribution $P$ and the realized outcome $y$. A scoring rule is **proper** if:

$$\mathbb{E}_{Y \sim Q}[S(Q, Y)] \leq \mathbb{E}_{Y \sim Q}[S(P, Y)]$$

for all distributions $P$ and $Q$, with equality only when $P = Q$.

### Brier Score

For binary outcomes, the Brier Score is:

$$BS = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

where $p_i$ is the predicted probability and $y_i \in \{0, 1\}$ is the actual outcome.

**Properties:**
- Strictly proper scoring rule
- Range: [0, 1] (lower is better)
- Decomposable into reliability, resolution, and uncertainty

### Logarithmic Score

$$LS = -\frac{1}{n}\sum_{i=1}^{n}\log(p_i(y_i))$$

where $p_i(y_i)$ is the predicted probability of the actual outcome.

**Properties:**
- Strictly proper
- Information-theoretic interpretation (log-likelihood)
- Sensitive to extreme probabilities

### Continuous Ranked Probability Score (CRPS)

For continuous outcomes:

$$CRPS(P, y) = \int_{-\infty}^{\infty}(F(x) - \mathbb{1}_{x \geq y})^2 dx$$

where $F$ is the cumulative distribution function of the forecast.

**Properties:**
- Proper scoring rule
- Generalizes MAE for probabilistic forecasts
- Can be decomposed into reliability and sharpness components

## 2. Forecast Combination

### Weighted Averaging

Given $M$ forecasts $\hat{y}_1, \ldots, \hat{y}_M$, the combined forecast is:

$$\hat{y}_c = \sum_{m=1}^{M} w_m \hat{y}_m$$

where $\sum_{m=1}^{M} w_m = 1$ and $w_m \geq 0$.

### Optimal Weights

Under mean squared error, optimal weights minimize:

$$E\left[\left(y - \sum_{m=1}^{M} w_m \hat{y}_m\right)^2\right]$$

This leads to:

$$\mathbf{w}^* = (\mathbf{\Sigma}^{-1}\mathbf{1}) / (\mathbf{1}'\mathbf{\Sigma}^{-1}\mathbf{1})$$

where $\mathbf{\Sigma}$ is the forecast error covariance matrix.

### Stacking (Meta-Learning)

Stacking uses a meta-model to learn optimal combination:

$$\hat{y}_c = f(\hat{y}_1, \ldots, \hat{y}_M; \boldsymbol{\theta})$$

where $f$ is typically a linear regression or more complex model.

## 3. Calibration

### Definition

A forecast is **calibrated** if:

$$P(Y = 1 | P = p) = p$$

for all probability levels $p$.

### Expected Calibration Error (ECE)

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

where $B_m$ are bins, $\text{acc}(B_m)$ is accuracy, and $\text{conf}(B_m)$ is average confidence.

### Isotonic Regression

Finds monotonic function $g$ that minimizes:

$$\sum_{i=1}^{n} (g(p_i) - y_i)^2$$

subject to $g(p_i) \leq g(p_j)$ if $p_i \leq p_j$.

### Platt Scaling

Fits logistic regression:

$$P(Y = 1 | f) = \frac{1}{1 + \exp(Af + B)}$$

where $f$ is the raw forecast and $A, B$ are learned parameters.

## 4. Time Series Models

### ARIMA(p,d,q)

An ARIMA model is:

$$\phi(B)(1-B)^d y_t = \theta(B)\epsilon_t$$

where:
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR component)
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA component)
- $d$ is the differencing order
- $B$ is the backshift operator

### Vector Autoregression (VAR)

A VAR(p) model:

$$\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{\Phi}_i \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$$

where $\mathbf{y}_t$ is a $k \times 1$ vector of variables.

**Impulse Response Function:**

$$IRF(h) = \frac{\partial y_{t+h}}{\partial \epsilon_t}$$

measures the effect of a one-unit shock.

### Cointegration and VECM

If variables are cointegrated, use Vector Error Correction Model:

$$\Delta \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}'\mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \mathbf{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$$

where $\boldsymbol{\beta}$ contains cointegrating vectors.

## 5. Bayesian Forecasting

### Posterior Predictive Distribution

$$p(y_{T+h} | \mathbf{y}_{1:T}) = \int p(y_{T+h} | \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathbf{y}_{1:T}) d\boldsymbol{\theta}$$

### Gaussian Process

A GP is defined by mean function $\mu(\mathbf{x})$ and covariance function $k(\mathbf{x}, \mathbf{x}')$:

$$f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

Common covariance functions:
- **RBF**: $k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\ell^2}\right)$
- **Matern**: Various smoothness levels

## 6. Statistical Tests

### Diebold-Mariano Test

Tests $H_0: E[d_t] = 0$ where $d_t$ is the loss differential:

$$d_t = L(e_{1t}) - L(e_{2t})$$

Test statistic:

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \sim N(0,1)$$

### Ljung-Box Test

Tests for autocorrelation in residuals:

$$Q = n(n+2)\sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(h)$$

### Augmented Dickey-Fuller Test

Tests for unit root in time series:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

Tests $H_0: \gamma = 0$ (unit root present).

## 7. Uncertainty Quantification

### Bootstrap

Resample with replacement to estimate sampling distribution:

$$\hat{F}_n^*(x) = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}_{\hat{\theta}_b^* \leq x}$$

### Conformal Prediction

Provides distribution-free prediction intervals with coverage guarantee:

$$P(y_{n+1} \in C_{1-\alpha}(X_{n+1})) \geq 1 - \alpha$$

## 8. Information Criteria

### Akaike Information Criterion (AIC)

$$AIC = -2\log L + 2k$$

where $L$ is likelihood and $k$ is number of parameters.

### Bayesian Information Criterion (BIC)

$$BIC = -2\log L + k\log n$$

Penalizes complexity more than AIC.

### Widely Applicable Information Criterion (WAIC)

Bayesian alternative:

$$WAIC = -2\sum_{i=1}^{n} \log \mathbb{E}_{post}[p(y_i | \theta)] + 2\sum_{i=1}^{n} \text{Var}_{post}[\log p(y_i | \theta)]$$

## Further Reading

See [REFERENCES.md](REFERENCES.md) for complete bibliography and additional resources.

---

*This document provides theoretical foundations. For implementation details, see [API.md](API.md).*

