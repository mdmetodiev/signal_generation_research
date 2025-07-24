## Predicting Log Returns Using PCA Embeddings of Technical Indicators

This project explores whether behavioural patterns in commonly used technical indicators can be leveraged to forecast future market returns. The model uses Principal Component Analysis (PCA) to embed a feature space of indicators and then applies a linear regression to predict log returns after *N* hours.

### Motivation

Technical indicators (e.g., RSI, MACD, Bollinger Bands) are widely used by market participants in various contexts — short-period RSI for scalping, longer-period divergences for momentum reversal, etc. Their ubiquity suggests they may encode aggregate behavioural patterns in market activity.

This model attempts to learn relationships between indicator configurations and subsequent price movements, treating them as an implicit reflection of crowd behaviour.

---

### Methodology

- **Feature Engineering**: A set of technical indicators is computed over historical price data.
- **Dimensionality Reduction**: PCA is applied to project the feature matrix into a lower-dimensional latent space.
- **Return Prediction**: A linear regression model is trained to predict the log return after *N* time steps.
- **Signal Generation**: Predictions near the upper and lower quantiles are used to trigger long/short trading signals.
- **Performance Evaluation**:
  - **Monte-Carlo Permutation Test**: Assesses whether the in-sample Sharpe ratio could have arisen by chance. A quasi p-value of 0.002 was obtained.
  - **Nested Walk-Forward Validation**: Used to simulate out-of-sample performance under realistic assumptions. The strategy achieved a Sharpe ratio of 1.28, albeit with periods of underperformance.

---

### Usage

- The notebook [`crypto_alpha_research.ipynb`](./crypto_alpha_research.ipynb) provides a detailed walkthrough of the approach, analysis, and results.
- All code for preprocessing, modeling, and evaluation is available in this repo.

---

### Next Steps

- Investigate and adapt to periods of underperformance
- Perform walk-forward Monte Carlo permutation testing
- Explore nonlinear models in the PCA-embedded space

---

### Related Work

This approach builds on ideas explored in [Are technical trading rules profitable? Evidence from Bitcoin](https://www.sciencedirect.com/science/article/pii/S2405918818300928), among other sources.

