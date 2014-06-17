import numpy as np
from glmnet import ElasticNet, LogisticNet
from cv.cv_glmnet import CVGlmNet, _fit_and_score, _clone
from sklearn.externals.joblib import dump, load

display_bar = '-'*70

print display_bar
print "Fit an elastic net on some fake data"
print display_bar

X = np.random.randn(5000, 40)
w = np.random.randn(40)
w[20:] = 0
y = np.dot(X, w) + .2*np.random.randn(5000)

enet = ElasticNet(alpha=1)

#C = _clone(enet)
#dump(C, 'my_model.mdl')
#CC = load('my_model.mdl')

#print _fit_and_score(CC, X, y, range(0, 5000, 2), range(1, 5000, 2))

enet_cv = CVGlmNet(enet, n_jobs=3)
enet_cv.fit(X, y)
