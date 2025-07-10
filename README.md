# Using PCA embedding of technical indicators to predict price

 * PCA is used to reduce the dimensionality, and then a linear regressor is used to predict log return change in next N measurements. 
 * We notice that the regressor is mostly accurate at the quantiles. We use those as thresholds to generate signals for long and short positions. 
 * We perform a Monte-Carlo permutation test to verify our optimisation is not due to luck.
 * We use walk-forward validation to test out of sample performance.
 
## Usage

* Description of the model is provided in crypto_alpha_research.ipynb





