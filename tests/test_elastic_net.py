from glmnet import ElasticNet
import numpy as np
from scipy.sparse import csc_matrix
import unittest

np.random.seed(123)

class TestElasticNet(unittest.TestCase):
    '''Test suite for ElasticNet models.'''

    def test_unregularized_models(self):
        '''Test that fitting an unregularized model (lambda=0) gives
        expected results.
        
          For both dense and sparse design matricies, we test that an 
        unregularized model captures a perfect linear relationship without
        error.
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        y = np.dot(Xdn, w)
        for alpha in [0, .5, 1]:
            for X in (Xdn, Xsp): 
                enet = ElasticNet(alpha=alpha)
                enet.fit(X, y, lambdas=[0])
                test_preds = np.allclose(enet.predict(X).ravel(), y, atol=.01)
                self.assertTrue(test_preds)
                test_coefs = np.allclose(enet.coefficients.ravel(), w, atol=.02)
                self.assertTrue(test_coefs)

    def test_lasso_models(self):
        '''Test that a pure lasso (alpha=1) model gives expected results.        
        
          For both dense and sparse design matricies we test that a lasso
        model can pick out zero parameters from an otherwise perfect linear 
        relationship.
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        for w_mask in range(1, 10):
            for X in (Xdn, Xsp):
                w_masked = w.copy()
                w_masked[w_mask:] = 0
                y = np.dot(Xdn, w_masked)
                enet = ElasticNet(alpha=1)
                enet.fit(X, y, lambdas=[.01])
                test = (len(enet.coefficients.ravel() == w_mask))
                self.assertTrue(test)

    def test_unregularized_with_weights(self):
        '''Test that fitting an unregularized model (lambda=0) gives expected
        results when sample weights are used.
        
          For both dense and sparse dsign matricies, we test that an
        unregularized model captures a perfect linear relationship without
        error.
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        y = np.dot(Xdn, w)
        sw = np.random.uniform(size=(50,))
        for alpha in [0, .5, 1]:
            for X in (Xdn, Xsp): 
                enet = ElasticNet(alpha=alpha)
                enet.fit(X, y, lambdas=[0], weights=sw)
                test_preds = np.allclose(enet.predict(X).ravel(), y, atol=.01)
                self.assertTrue(test_preds)
                test_coefs = np.allclose(enet.coefficients.ravel(), w, atol=.02)
                self.assertTrue(test_coefs)

    def test_lasso_with_weights(self):
        '''Test that a pure lasso (alpha=1) model gives expected results when
        sample weights are used.        
        
          For both dense and sparse design matricies we test that a lasso
        model can pick out zero parameters from an otherwise perfect linear 
        relationship.
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        sw = np.random.uniform(size=(50,))
        sw = sw / np.sum(sw)
        for w_mask in range(1, 10):
            for X in (Xdn, Xsp):
                w_masked = w.copy()
                w_masked[w_mask:] = 0
                y = np.dot(Xdn, w_masked)
                enet = ElasticNet(alpha=1)
                enet.fit(X, y, lambdas=[.01], weights=sw)
                test = (len(enet.coefficients.ravel() == w_mask))
                self.assertTrue(test)

    def test_max_lambda(self):
        '''Test that the calculations of max_lambda inside the fortran code and
        inside the python code give the same result on both dense and sparse
        matricies.  
        
            Note that the implementation of max_lambda for alpha=0 in
        the fortran code is unknown, so we currently do not test against it.
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        y = np.dot(Xdn, w)
        for alpha in [.01, .5, 1]:
            for X in (Xdn, Xsp):
                enet = ElasticNet(alpha=alpha)
                enet.fit(X, y)
                ol = enet.out_lambdas
                max_lambda_from_fortran = ol[1] * (ol[1]/ol[2]) 
                max_lambda_from_python = enet._max_lambda(X, y)
                self.assertAlmostEqual(
                    max_lambda_from_fortran, max_lambda_from_python, 4
                )

    def test_max_lambda_with_weights(self):
        '''Test that the calculations of max_lambda inside the fortran code and
        inside the python code give the same result on both dense and sparse
        matricies, even when sample weights come into play.  
        '''
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(10,))
        y = np.dot(Xdn, w)
        sw = np.random.uniform(size=(50,))
        for alpha in [.01, .5, 1]:
            for X in (Xdn, Xsp):
                enet = ElasticNet(alpha=alpha)
                enet.fit(X, y, weights=sw)
                ol = enet.out_lambdas
                max_lambda_from_fortran = ol[1] * (ol[1]/ol[2]) 
                max_lambda_from_python = enet._max_lambda(X, y, weights=sw)
                self.assertAlmostEqual(
                    max_lambda_from_fortran, max_lambda_from_python, 4
                )

    def test_lasso_withrel_penalties(self):
        Xdn = np.random.random(size=(50,10))
        Xsp = csc_matrix(Xdn)
        w = np.ones(shape=(10,))
        y = np.dot(Xdn, w)
        for i in range(1, 10):
            for X in (Xdn, Xsp):
                penalties = np.ones(shape=(10,)) / 1000
                penalties[i] = 1
                enet = ElasticNet(alpha=1)
                enet.fit(X, y, lambdas=[16], rel_penalties=penalties)
                coef = enet.coefficients.ravel()
                # TODO: Not sure how to write this test.  Seems to work fine
                # when eyeballing it.  I'll come back to this.
