# Using PCA embedding of technical indicators to predict price

The model uses commonly known technical indicators to predict log returns after N hours. These indicators are widely used by market participants to respond to different conditions. For example, a short-period RSI might be used for scalping, while longer periods might be employed to detect divergences. Because such indicators are embedded in a broad spectrum of trading behaviours, it’s plausible that they encode the collective actions of traders. My model aims to identify and exploit these behavioural patterns by learning relationships between indicator configurations and future return

 * PCA is used to reduce the dimensionality, and then a linear regressor is used to predict log return change in next N measurements. 
 * We notice that the regressor is mostly accurate at the quantiles. We use those as thresholds to generate signals for long and short positions. 
 * We perform a Monte-Carlo permutation test to verify our optimisation is not due to luck.
 * We use walk-forward validation to test out of sample performance.
 
## Usage

* Description of the model is provided in crypto_alpha_research.ipynb





