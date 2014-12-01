from glmnet import LogisticNet
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import unittest

np.random.seed(123)

class TestLogisticNet(unittest.TestCase):
    '''Test suite for LogisticNet models.'''

    def test_unregularized_models(self):
        Xdn = np.random.uniform(-1, 1, size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.uniform(-1, 1, size=(10,))
        y = (np.dot(Xdn, w) >= 0).astype(int)
        for alpha in [0, .5, 1]:
            for X in (Xdn, Xsp):
                lnet = LogisticNet(alpha=alpha)
                lnet.fit(X, y, lambdas=[0])
                preds = (lnet.predict(X) >= .5).astype(int)
                self.assertTrue(np.all(y == preds))

    def test_lasso_models(self):
        Xdn = np.random.uniform(-1, 1, size=(500,10))
        Xsp = csc_matrix(Xdn)
        w = (np.random.uniform(-1, 1, size=(10,)) >= 0).astype(int) - .5
        for w_mask in range(1, 10):
            for X in (Xdn, Xsp):
                w_masked = w.copy()
                w_masked[w_mask:] = 0
                y = (np.dot(Xdn, w_masked) >= 0).astype(int)
                lnet = LogisticNet(alpha=1)
                lnet.fit(X, y, lambdas=[.01])
                lc = lnet._coefficients
                self.assertTrue(
                   np.sum(np.abs(lc / np.max(np.abs(lc))) > .05) == w_mask
                )

    def test_max_lambda(self):
        Xdn = np.random.uniform(-1, 1, size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.uniform(-1, 1, size=(10,))
        y = (np.dot(Xdn, w) >= 0).astype(int)
        for X in (Xdn, Xsp):
            for alpha in [.01, .5, 1]:
                lnet = LogisticNet(alpha=alpha)
                lnet.fit(X, y)
                ol = lnet.out_lambdas
                print ol
                max_lambda_from_fortran = ol[1] * (ol[1]/ol[2]) 
                print max_lambda_from_fortran 
                max_lambda_from_python = lnet._max_lambda(X, y)
                print max_lambda_from_python
                self.assertAlmostEqual(
                    max_lambda_from_fortran, max_lambda_from_python, 4
                )
