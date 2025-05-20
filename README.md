# Using a random forest regressor to predict price

PCA is used to reduce the dimensionality, and then a random forest regressor is used to predict log return change in next N measurements. We notice that the regressor is mostly accurate at the quantiles. We use those as thresholds to generate signals for long and short positions. We use walk-forward validation to test out of sample performance.
 

## Usage

* Description of the model is provided in notebook.ipynb
* Model training: Import learn.py to walk forward the RF model; additionaly, a model.pickle object can be saved. This is a dictionary holding the upper/lower quantile values, PCA object and regressor object. This can be imported later for live testing
* Live testing (still buggy): to test the model in a live scenario, run live_monitoring.py. It will fetch hourly data from coinbase and apply the model. On interupt, it will dump out all the recorded data in a model_performance_data file (pickled dictionary with some stats).


## TODOs:

- [x]  add a jupyter notebook detailing model
- [x]  add slippage/costs 
- [ ]  train a classifier that can find signals that are worth trading
- [ ]  use Monte-Carlo walk forward (when I find a good model) to further test that I am not overfitting to noise. I've written the code but there is no point testing it with current model.



