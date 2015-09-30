import numpy as np
from ..glmnet import ElasticNet
from sklearn.datasets import make_regression

display_bar = '-'*70

X, y = make_regression(
    n_samples = 5000,
    n_features = 100,
    n_informative = 30,
    effective_rank = 40,
    noise = .1,
)

print display_bar
print "Fit an elastic net on some fake data"
print display_bar

enet = ElasticNet(alpha=.025)
enet.fit(X, y)

print enet

print display_bar
print "Predictions vs. actuals for the last elastic net model:"
print display_bar

preds = enet.predict(X)
print y[:10]
print preds[:10,np.shape(preds)[1]-1]

enet.plot_paths()
