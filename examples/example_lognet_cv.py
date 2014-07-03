import numpy as np
from ..glmnet import LogisticNet
from sklearn.datasets import make_regression
from ..cv.cv_glmnet import CVGlmNet

display_bar = '-'*70

X = np.random.randn(5000, 50)
b = 3*np.random.randn(50)
b[20:] = 0
p = 1/(1 + np.exp(np.dot(X, b)))
u = np.random.uniform(size=5000)
y = np.int64(p > u)
# Flip some random bits
bits = np.random.binomial(1, .25, size=5000)
y = (1 - bits)*y + bits*(1 - y)

print display_bar
print "Cross validate an elastic net on some fake data"
print display_bar

lognet = LogisticNet(alpha=.1)
lognet_cv = CVGlmNet(lognet, n_jobs=1)
lognet_cv.fit(X, y)

print display_bar
print lognet_cv.base_estimator

lognet_cv.plot_oof_devs()
