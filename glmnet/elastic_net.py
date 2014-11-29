import numpy as np
from scipy.sparse import issparse
from sklearn import preprocessing
import _glmnet
from glmnet import GlmNet

class ElasticNet(GlmNet):
    '''The elastic net: a multivariate linear model with both L1 and L2
    regularizers.

      This class implements the elastic net class of predictive models. These
    models combine the classical ridge and lasso regularizers into a combined
    penalty to the sum of squared residuals loss function.  More specifically,
    the loss function minimized by this model is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            RSS(\beta_0, \beta_1, ..., \beta_n; X, y) + 
            \lambda * ((\alpha - 1)/2 * | \beta |_2 + \alpha * | \beta |_1)

    where RSS is the usual residual sum of squares:

      RSS(\beta_0, \beta_1, ..., \beta_n; X, y) = sum((\beta_i * X_i - y_i)^2)
    '''

    # TODO: Implement offsets.
    def fit(self, X, y,
            lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None):
        '''Fit an elastic net model.

        Arguments: 

          * X: The model matrix.  A n_obs * n_preds array.
          * y: The response.  A n_obs array.

        Optional Arguments:
          
          * lambdas: 
              A user supplied list of lambdas, an elastic net will be fit for
              each lambda supplied.  If no array is passed, glmnet will generate
              its own array of lambdas equally spaced on a logaritmic scale 
              between \lambda_max and \lambda_min.
          * weights: 
               An n_obs array. Sample weights.
          * rel_penalties: 
              An n_preds array. Relative panalty weights for the covariates.  If
              none is passed, all covariates are penalized equally.  If an array
              is passed, then a zero indicates an unpenalized parameter, and a 1
              a fully penalized parameter.  Otherwise all covaraites recieve an
              equal penalty.
          * excl_preds: 
              An n_preds array, used to exclude covaraites from the model. To
              exclude predictors, pass an array with a 1 in the first position,
              then a 1 in the i+1st position excludes the ith covaraite from
              model fitting.  If no array is passed, all covaraites in X are 
              included in the model.
          * box_constraints: 
              An array with dimension 2 * n_obs. Interval constraints on the fit
              coefficients.  The (0, i) entry is a lower bound on the ith
              covariate, and the (1, i) entry is an upper bound.  These must 
              satisfy lower_bound <= 0 <= upper_bound.  If no array is passed,
              no box constraintes are allied to the parameters.

        After fitting, the following attributes are set:
        
        Private attributes:

          * _n_fit_obs:
              The number of rows in the model matrix X.
          * _n_fit_params:
              The number of columns in the model matrix X.
          * _out_n_lambdas: 
              The number of lambdas associated with non-zero models (i.e.
              models with at least one none zero parameter estiamte) after
              fitting; for large enough lambda the models will become zero in
              the presense of an L1 regularizer.
          * _intecepts: 
              A one dimensional array containing the intercept estiamtes for
              each value of lambda.  See the intercepts (no underscore) 
              property for a public version.
          * _comp_coef: 
              The fit parameter estiamtes in a compressed form.  This is a
              matrix with each row giving the estimates for a single
              coefficient for various values of \lambda.  The order of the rows
              does not correspond to the order of the coefficents as given in
              the design matrix used to fit the model, this association is
              given by the _p_comp_coef attribute.  Only estaimtes that are
              non-zero for some lambda are reported.
          * _p_comp_coef: 
              A one dimensional integer array associating the coefficients in
              _comp_coef to columns in the model matrix. 
          * _indicies: 
              The same information as _p_comp_coef, but zero indexed to be
              compatable with numpy arrays.
          * _n_comp_coef: 
              The number of parameter estimates that are non-zero for some
              value of lambda.
          * _n_passes: 
              The total number of passes over the data used to fit the model.
          * _error_flag: 
              Error flag from the fortran code.

        Public Attributes:

          * r_sqs: 
              An array of length _out_n_lambdas containing the r-squared
              statistic for each model.
          * out_lambdas: 
              An array containing the lambda values associated with each fit
              model.
        '''
        self._check_if_unfit()
        # Convert to arrays if native python objects
        try:
            if not issparse(X):
                X = np.asanyarray(X)
            y = np.asanyarray(y)
        except ValueError:
            raise ValueError("X and y must be either numpy arrays, or "
                             "convertable to numpy arrays."
                  )
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if not issparse(X) and np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # Make a copy if we are not able to overwrite y with its standardized
        # version.
        if not self.overwrite_targ_ok:
            y = y.copy()
        # Validate all the inputs:
        self._validate_matrix(X)
        self._validate_inputs(X, y)
        self._validate_lambdas(X, y, lambdas)
        self._validate_weights(X, y, weights)
        self._validate_rel_penalties(X, y, rel_penalties)
        self._validate_excl_preds(X, y, excl_preds)
        self._validate_box_constraints(X, y, box_constraints)
        # Setup is complete, call into the extension module.
        if not issparse(X):
            (self._out_n_lambdas,
             self._intercepts,
             self._comp_coef,
             self._p_comp_coef,
             self._n_comp_coef,
             self.r_sqs,
             self.out_lambdas,
             self._n_passes,
             self._error_flag) = _glmnet.elnet(
                                     self.alpha, 
                                     X, 
                                     y, 
                                     self.weights, 
                                     self.excl_preds, 
                                     self.rel_penalties,
                                     self.box_constraints,
                                     self.max_vars_all, 
                                     self.frac_lg_lambda, 
                                     self.lambdas, 
                                     self.threshold, 
                                     nlam=self.n_lambdas
                                 )
        else:
            X.sort_indices()
            # Fortran arrays are 1 indexed.
            ind_ptrs = X.indptr + 1
            indices = X.indices + 1
            # Call
            (self._out_n_lambdas,
            self._intercepts,
            self._comp_coef,
            self._p_comp_coef,
            self._n_comp_coef,
            self.r_sqs,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.spelnet(
                                    self.alpha, 
                                    X.shape[0],
                                    X.shape[1],
                                    X.data, 
                                    ind_ptrs, 
                                    indices,
                                    y,
                                    self.weights, 
                                    self.excl_preds, 
                                    self.rel_penalties,
                                    self.box_constraints,
                                    self.max_vars_all, 
                                    self.frac_lg_lambda, 
                                    self.lambdas, 
                                    self.threshold, 
                                    nlam=self.n_lambdas
                                )
        self._check_errors()
        # Keep some model metadata
        self._n_fit_obs, self._n_fit_params = X.shape
        # The indexes into the predictor array are off by one due to fortran
        # convention differing from numpys, this make them indexes into the the
        # numpy array. 
        self._indicies = np.trim_zeros(self._p_comp_coef, 'b') - 1

    @property
    def coefficients(self):
        '''The fit model coefficients for each lambda.

          A _n_comp_coef * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        self._check_if_fit()
        return self._comp_coef[:np.max(self._n_comp_coef),
                               :self._out_n_lambdas
                ]

    def _max_lambda(self, X, y, weights=None):
        '''Return the maximum value of lambda useful for fitting, i.e. that
        which first forces all coefficients to zero.

          The calculation is derived from the discussion in "Regularization 
        Paths for Generalized Linear Models via Coordinate Descent".
        '''
        if issparse(X):
            return self._max_lambda_sparse(X, y, weights)
        else:
            return self._max_lambda_dense(X, y, weights)

    def _max_lambda_dense(self, X, y, weights=None):
        dot = self._get_dot(X)
        if weights is None:
            # Standardize X and then find the maximum dot product.
            normfac = X.shape[0]
            mu = X.sum(axis=0) / normfac
            mu2 = (X*X).sum(axis=0) / normfac
            X_scaled = (X - mu) / np.sqrt(mu2 - mu*mu)
            dots = dot(y, X_scaled)
        else:
            # Standardize X using the sample weights and then find the
            # maximum weighted dot product.
            y_wtd = y * weights
            mu = dot(weights, X)
            mu2 = dot(weights, X*X)
            X_scaled = (X - mu) / np.sqrt(mu2 - mu*mu)
            dots = dot(y_wtd, X_scaled)
            # Since we included weights in the dot product we do not need
            # to include the weight in the denominator.
            normfac = 1
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / (alpha * normfac) 

    def _max_lambda_sparse(self, X, y, weights=None):
        '''To preserve the sparsity, we must avoid explicitly subtracting out
        the mean of the columns.
        '''
        # Sparse dot
        dot = self._get_dot(X)
        # Calculate the dot product of y with X standardized, without
        # destorying the sparsity of X.  The calculations themselves do not
        # differ from the dense case.
        if weights is None:
            E = lambda M: np.asarray(M.sum(axis=0)).ravel() / M.shape[0]
            mu = E(X)
            mu_2 = E(X.multiply(X))
            sigma = np.sqrt(mu_2 - mu*mu)
            dots = 1/sigma * (dot(y, X) - mu * np.sum(y))
            normfac = X.shape[0]
        else:
            y_wtd = y*weights
            E = lambda M, wts: dot(wts, M)
            mu = E(X, weights)
            mu_2 = E(X.multiply(X), weights)
            sigma = np.sqrt(mu_2 - mu*mu)
            dots = 1/sigma * (dot(y_wtd, X) - mu * np.sum(y_wtd))
            normfac = 1
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / (alpha * normfac) 

    def deviance(self, X, y, weights = None):
        '''Calculate the normal deviance (i.e. sum of squared errors) for
        every lambda.  The model must already be fit to call this method.
        '''
        self._check_if_fit()
        if weights is not None and weights.shape[0] != X.shape[0]:
            raise ValueError("The weights vector must have the same length "
                             "as X."
                  )
        y_hat = self.predict(X)
        # Take the response y, and repeat it as a column to produce a matrix
        # of the same dimensions as y_hat
        y_stacked = np.tile(np.array([y]).transpose(), y_hat.shape[1])
        if weights is None:
            sq_residuals = (y_stacked - y_hat)**2
            normfac = X.shape[0]
        else:
            w_stacked = np.tile(np.array([weights]).transpose(),
                                y_hat.shape[1]
                        )
            sq_residuals = w_stacked * (y_stacked - y_hat)**2
            normfac = np.sum(weights)
        return np.apply_along_axis(np.sum, 0, sq_residuals) / normfac

    def predict(self, X):
        '''Produce model predictions from new data.'''
        return self._predict_lp(X)


    def plot_paths(self):
        self._plot_path('elastic') 

    def __str__(self):
        return self._str('elastic')

