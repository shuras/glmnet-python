from glmnet import ElasticNet
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import unittest

np.random.seed(123)

class TestElasticNet(unittest.TestCase):
    '''Test suite for ElasticNet models.'''

    def test_unregularized_models(self):
        '''Test that fitting an unregularized model (lambda=0) gives
        expected results for both dense and sparse model matricies.
        
          We test that an unregularized model captures a perfect linear
        relationship without error.  That is, the fit parameters equals the
        true coefficients.
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
                test_coefs = np.allclose(enet._coefficients.ravel(), w, atol=.02)
                self.assertTrue(test_coefs)

    def test_lasso_models(self):
        '''Test that a pure lasso (alpha=1) model gives expected results
        for both dense and sparse design matricies.        
        
          We test that the lasso model has the ability to pick out zero 
        parameters from a linear relationship.  To see this, we generate 
        linearly related data were some number of the coefficients are
        exactly zero, and make sure the lasso model can pick these out.
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
                test = (len(enet._coefficients.ravel() == w_mask))
                self.assertTrue(test)

    def test_ridge_models(self):
        '''Test that a pure ridge (alpha=0) model gives expected results
        for both dense and sparse matricies.

          We test that the ridge model, when fit on uncorrelated predictors,
        shrinks the parameter estiamtes uniformly.  To see this, we generate
        linearly related data with a correlation free model matrix, then test
        that the array of ratios of fit parameters to true coefficients is 
        a constant array.
        
        This test generates more samples than the others to guarentee that the
        data is sufficiently correlation free, otherwise the effect to be 
        measured does not occur.
        '''
        Xdn = np.random.random(size=(10000,3))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(3,))
        for X in (Xdn, Xsp):
            for lam in np.linspace(0, 1, 10):
                y = np.dot(Xdn, w)
                enet = ElasticNet(alpha=0)
                enet.fit(X, y, lambdas=[lam])
                ratios = enet._coefficients.ravel() / w
                norm_ratios = ratios / np.max(ratios)
                test = np.allclose(
                    norm_ratios, 1, atol=.05
                )
                self.assertTrue(test)

    def test_unregularized_with_weights(self):
        '''Test that fitting an unregularized model (lambda=0) gives expected
        results when sample weights are used.
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
                test_coefs = np.allclose(enet._coefficients.ravel(), w, atol=.02)
                self.assertTrue(test_coefs)

    def test_lasso_with_weights(self):
        '''Test that a pure lasso (alpha=1) model gives expected results when
        sample weights are used.        
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
                test = (len(enet._coefficients.ravel() == w_mask))
                self.assertTrue(test)

    def test_ridge_with_weights(self):
        '''Test that a pure ridge (alpha=0) model gives expected results
        for both dense and sparse matricies.
        '''
        Xdn = np.random.random(size=(10000,3))
        Xsp = csc_matrix(Xdn)
        w = np.random.random(size=(3,))
        sw = np.random.uniform(size=(10000,))
        sw = sw / np.sum(sw)
        for X in (Xdn, Xsp):
            for lam in np.linspace(0, 1, 10):
                y = np.dot(Xdn, w)
                enet = ElasticNet(alpha=0)
                enet.fit(X, y, lambdas=[lam], weights=sw)
                ratios = enet._coefficients.ravel() / w
                norm_ratios = ratios / np.max(ratios)
                test = np.allclose(
                    norm_ratios, 1, atol=.05
                )
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

    def test_validate_matrix(self):
        '''Test the _validate_matrix method.'''
        Xdn = np.random.random(size=(50,10))
        enet = ElasticNet(alpha=.5)
        # Invalid use:
        #   Passing in a sparse matrix in the incorrect format.
        with self.assertRaises(ValueError):
            Xsp = csr_matrix(Xdn)
            enet._validate_matrix(Xsp)
        # Valid use:
        #   Passing in a matrix in compressed sparse column format.
        Xsp = csc_matrix(Xdn)
        enet._validate_matrix(Xsp)

    def test_validate_inputs(self):
        X = np.random.random(size=(50,10))
        w = np.random.random(size=(10,))
        enet = ElasticNet(alpha=.5)
        # Invalid Use:
        #    Passing in a y that is too short.
        with self.assertRaises(ValueError):
            yprime = np.random.random(size=(49,))
            enet._validate_inputs(X, yprime)
        # Invalid use:
        #    Passing in a y that matches the wrong dimenstion of X.
        with self.assertRaises(ValueError):
            yprime = np.random.random(size=(10,))
            enet._validate_inputs(X, yprime)
        # Valid Use:
        #    Passing in a y of the correct dimension.
        yprime = np.random.random(size=(50,))
        enet._validate_inputs(X, yprime)

    def test_validate_weights(self):
        X = np.random.random(size=(50,10))
        w = np.random.random(size=(10,))
        y = np.dot(X, w)
        enet = ElasticNet(alpha=.5)
        # Invalid use
        #    Passing in a sample weight vector that is too short.
        with self.assertRaises(ValueError):
            sw = np.ones(shape=(49,))
            enet._validate_weights(X, y, weights=sw)
        # Invalid use:
        #    Passing in a weight vector that matches the wrong dimenstion of X.
        with self.assertRaises(ValueError):
            sw = np.ones(shape=(10,))
            enet._validate_weights(X, y, weights=sw)
        # Invalid use:
        #    Passing in a weight vector containing a negative entry. 
        with self.assertRaises(ValueError):
            sw = np.ones(shape=(50,))
            sw[25] = -1
            enet._validate_weights(X, y, weights=sw)
        # Valid Use:
        #    Weight vector of the correct dimension with all non-negative 
        #    entries.
        sw = np.ones(shape=(50,))
        enet._validate_weights(X, y, weights=sw)

    def test_validate_rel_penalties(self):
        X = np.random.random(size=(50,10))
        w = np.random.random(size=(10,))
        y = np.dot(X, w)
        enet = ElasticNet(alpha=.5)
        # Invalid use
        #    Passing in a rel_penalties vector that is too short.
        with self.assertRaises(ValueError):
            rel_pens = np.ones(shape=(9,))
            enet._validate_rel_penalties(X, y, rel_penalties=rel_pens)
        # Invalid use:
        #    Passing in a rel_penalties that matches the wrong dimenstion of X.
        with self.assertRaises(ValueError):
            rel_pens = np.ones(shape=(50,))
            enet._validate_rel_penalties(X, y, rel_penalties=rel_pens)
        # Invalid use:
        #    Passing in a rel_penalties containing a negative entry. 
        with self.assertRaises(ValueError):
            rel_pens = np.ones(shape=(10,))
            rel_pens[5] = -1
            enet._validate_rel_penalties(X, y, rel_penalties=rel_pens)
        # Valid use:
        #    Rel_penalties has the correct dimenstion with all non-negative
        #    entries.
        rel_pens = np.ones(shape=(10,))
        rel_pens[5] = 0 
        enet._validate_rel_penalties(X, y, rel_penalties=rel_pens)

    def test_validate_excl_preds(self):
        X = np.random.random(size=(50,10))
        w = np.random.random(size=(10,))
        y = np.dot(X, w)
        enet = ElasticNet(alpha=.5)
        # Invalid use
        #    Passing in a excl_preds array that is to long.
        with self.assertRaises(ValueError):
            excl_preds = np.ones(shape=(12,))
            enet._validate_excl_preds(X, y, excl_preds=excl_preds)
        # Invalid use
        #    Alltempt to exclude a predictor out of range, i.e. that does
        #    not exist.
        with self.assertRaises(ValueError):
            excl_preds = np.ones(shape=(11,))
            excl_preds[0] = 1
            excl_preds[5] = 10
            enet._validate_excl_preds(X, y, excl_preds=excl_preds)
        # Valid use 
        #    Exclude some in range predictors.
        excl_preds = np.array([1, 2, 4, 6, 8])
        enet._validate_excl_preds(X, y, excl_preds=excl_preds)

    def test_validate_box_constraints(self):
        X = np.random.random(size=(50,10))
        w = np.random.random(size=(10,))
        y = np.dot(X, w)
        enet = ElasticNet(alpha=.5)
        # Invalid use
        #   Second dimension matches incorrect dimension of X.
        with self.assertRaises(ValueError):
           box_const = np.empty(shape=(2,50))
           enet._validate_box_constraints(X, y, box_constraints=box_const)
        # Invalid use
        #    Incorrect first dimension.
        with self.assertRaises(ValueError):
           box_const = np.empty(shape=(1,10))
           enet._validate_box_constraints(X, y, box_constraints=box_const)
        # Invalid use
        #    Transpose of a correct use.
        with self.assertRaises(ValueError):
           box_const = np.empty(shape=(10,2))
           enet._validate_box_constraints(X, y, box_constraints=box_const)
        # Invalid use
        #    Positive lower bound in constraint.
        with self.assertRaises(ValueError):
           box_const = np.empty(shape=(2,10))
           box_const[0,:] = -1; box_const[1,:] = 1
           box_const[0,5] = 1
           enet._validate_box_constraints(X, y, box_constraints=box_const)
        # Invalid use
        #    Negative upper bound in constraint.
        with self.assertRaises(ValueError):
           box_const = np.empty(shape=(2,10))
           box_const[0,:] = -1; box_const[1,:] = 1
           box_const[1,5] = -1
           enet._validate_box_constraints(X, y, box_constraints=box_const)
        # Valid use
        #     Impose some box constraints.
        box_const = np.empty(shape=(2,10))
        box_const[0,:] = -1; box_const[1,:] = 1
        enet._validate_box_constraints(X, y, box_constraints=box_const)
