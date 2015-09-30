import numpy as np
from ..glmnet import ElasticNet
from sklearn.datasets import make_regression
from ..cv.cv_glmnet import CVGlmNet

display_bar = '-'*70

X, y = make_regression(
    n_samples = 5000,
    n_features = 100,
    n_informative = 40,
    effective_rank = 30,
    noise = 8,
)
w = np.random.uniform(size=5000)

print display_bar
print "Cross validate an elastic net on some fake data"
print display_bar

enet = ElasticNet(alpha=.1)
enet_cv = CVGlmNet(enet, n_jobs=1)
enet_cv.fit(X, y, weights=w)

enet_cv.plot_oof_devs()
