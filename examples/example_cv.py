import numpy as np
from ..glmnet import ElasticNet
from sklearn.datasets import make_regression
from ..cv.cv_glmnet import CVGlmNet

display_bar = '-'*70

X, y = make_regression(
    n_samples = 5000,
    n_features = 100,
    n_informative = 20,
    effective_rank = 10,
    noise = .1,
)

print display_bar
print "Cross validate an elastic net on some fake data"
print display_bar

enet = ElasticNet(alpha=.1)
enet_cv = CVGlmNet(enet, n_jobs=1)
enet_cv.fit(X, y)

print display_bar
print enet_cv.base_estimator
