import numpy as np
from ..glmnet import LogisticNet
from sklearn.datasets import make_classification

display_bar = '-'*70

X = np.random.randn(5000, 50)
b = 3*np.random.randn(50)
b[20:] = 0
p = 1/(1 + np.exp(np.dot(X, b)))
u = np.random.uniform(size=5000)
y = np.int64(p > u)
# Flip some random bits
bits = np.random.binomial(1, .025, size=5000)
y = (1 - bits)*y + bits*(1 - y)

print display_bar
print "Fit a logistic net on some fake data"
print display_bar

lognet = LogisticNet(alpha=1)
lognet.fit(X, y)

print lognet

print display_bar
print "Predictions vs. actuals for the last logistic net model:"
print display_bar

preds = lognet.predict(X)
print preds[:10,np.shape(preds)[1]-1]
print p[:10]

lognet.plot_path()
