import numpy as np
from scipy.sparse import issparse
from sklearn import preprocessing
import _glmnet
from glmnet import GlmNet

class LogisticNet(GlmNet):
    '''The logistic net: a multivariate logistic model with both L1 and L2
    regularizers.

      This class implements the logistic net class of predictive models. These
    models combine the classical ridge and lasso regularizers into a combined
    penalty.  More specifically, the loss function minimized by this model is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            BinDev(\beta_0, \beta_1, ..., \beta_n; X, y) + 
            \lambda * ( (\alpha - 1)/2 * | \beta |_2 + \alpha * | \beta |_1 )

    where BinDev is the usual binomial deviance:

      BinDev(\beta_0, \beta_1, ..., \beta_n; X, y) =
       -2*sum( y*log(1 - p) + (1 - y)*log(1 - p) )

    where p is formed by applying the usual logistic function to a linear
    predictor.
    '''

    def fit(self, X, y,
            lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None, offsets=None):
        '''Fit a logistic or multinomial net model.

        Arguments:

          * X: The predictors.  A n_obs * n_preds array.

          * y: The response.  This method accepts the predictors in two
            differnt configurations:

            - An n_obs * n_classes array.  In this case, each column in y must
              be a boolean flag indicating whether the observation is or is not
              of this class.
            - An n_obs array.  In this case the array must contain a discrete
              number of values, and is converted into the previous form before
              being passed to the model.

        Optional Arguments:

          * lambdas: A user supplied list of lambdas, an elastic net will be 
            fit for each lambda supplied.  If no array is passed, glmnet 
            will generate its own array of lambdas.
          * weights: An n_obs array. Observation weights.
          * rel_penalties: An n_preds array. Relative panalty weights for the
            covariates.  If none is passed, all covariates are penalized 
            equally.  If an array is passed, then a zero indicates an 
            unpenalized parameter, and a 1 a fully penalized parameter.
          * excl_preds: An n_preds array, used to exclude covaraites from 
            the model. To exclude predictors, pass an array with a 1 in the 
            first position, then a 1 in the i+1st position excludes the ith 
            covaraite from model fitting.
          * box_constraints: An array with dimension 2 * n_obs. Interval 
            constraints on the fit coefficients.  The (0, i) entry
            is a lower bound on the ith covariate, and the (1, i) entry is
            an upper bound.
          * offsets: A n_preds * n_classes array. Used as initial offsets for
            the model fitting. 

        After fitting, the following attributes are set:
        
        Private attributes:

          * _out_n_lambdas: The number of fit lambdas associated with non-zero
            models; for large enough lambdas the models will become zero in the
            presense of an L1 regularizer.
          * _comp_coef: The fit coefficients in a compressed form.  Only
            coefficients that are non-zero for some lambda are reported, and the
            associated between these parameters and the predictors are given by
            the _p_comp_coef attribute.
          * _p_comp_coef: An array associating the coefficients in _comp_coef to
            columns in the predictor array. 
          * _indicies: The same information as _p_comp_coef, but zero indexed to
            be compatable with numpy arrays.
          * _n_comp_coef: The number of coefficients that are non-zero for some
            value of lambda.
          * _n_passes: The total number of passes over the data used to fit the
            model.
          * _error_flag: Error flag from the fortran code.

        Public Attributes:
          
          * null_dev: The devaince of the null model.
          * exp_dev: The devaince explained by the model.
          * out_lambdas: An array containing the lambda values associated with
            each fit model.
        '''
        if weights is not None:
            raise ValueError("LogisticNet cannot be fit with weights.")
        # Convert to arrays is native python objects
        try:
            if not issparse(X):
                X = np.asanyarray(X)
            y = np.asanyarray(y)
        except ValueError:
            raise ValueError("X and y must be wither numpy arrays, or "
                             "convertable to numpy arrays."
                  )
        # Fortran expects an n_obs * n_classes array for y.  If a one 
        # dimensional array is passed, we construct an appropriate widening. 
        y = np.asanyarray(y)
        if len(y.shape) == 1:
            self.logistic = True
            y_classes = np.unique(y)
            y = np.float64(np.column_stack(y == c for c in y_classes))
        else:
            self.logistic = False
        # Count the number of classes in y.
        y_level_count = y.shape[1]
        # Two class predictions are handled as a special case, as is usual 
        # with logistic models
        if y_level_count == 2:
            self.y_level_count = np.array([1])
        else:
            self.y_level_count = np.array([y_level_count])
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if not issparse(X) and np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # The target array will usually be overwritten with its standardized
        # version, if this is not ok, we should copy.
        if np.isfortran(y) and not self.overwrite_targ_ok:
            y = y.copy(order='F')
        # Validate all the inputs:
        self._validate_matrix(X)
        self._validate_inputs(X, y)
        self._validate_lambdas(X, y, lambdas)
        self._validate_weights(X, y, weights)
        self._validate_rel_penalties(X, y, rel_penalties)
        self._validate_excl_preds(X, y, excl_preds)
        self._validate_box_constraints(X, y, box_constraints)
        self._validate_offsets(X, y, offsets)
        # Setup is complete, call the wrapper
        if not issparse(X):
            (self._out_n_lambdas,
            self._intercepts,
            self._comp_coef,
            self._p_comp_coef,
            self._n_comp_coef,
            self.null_dev,
            self.exp_dev,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.lognet(
                                    self.alpha, 
                                    self.y_level_count,
                                    X,
                                    y, 
                                    self.offsets,
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
            ind_ptrs = X.indptr + 1
            indices = X.indices + 1
            (self._out_n_lambdas,
            self._intercepts,
            self._comp_coef,
            self._p_comp_coef,
            self._n_comp_coef,
            self.null_dev,
            self.exp_dev,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.splognet(
                                    self.alpha, 
                                    X.shape[0],
                                    X.shape[1],
                                    self.y_level_count,
                                    X.data,
                                    ind_ptrs,
                                    indices,
                                    y, 
                                    self.offsets,
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
        # convention, fix it up.
        self._indicies = np.trim_zeros(self._p_comp_coef, 'b') - 1

    def _validate_offsets(self, X, y, offsets):
        '''If no explicit offset was setup, we assume a zero offset.'''
        if offsets is None:
            self.offsets = np.zeros((X.shape[0], y.shape[1]), order='F')
        else:
            self.offsets = offsets
        if (self.offsets.shape[0] != X.shape[0] or
            self.offsets.shape[1] != y.shape[1]):
            raise ValueError("Offsets must share its first dimenesion with X "
                             "and its second dimenstion must be the number of "
                             "response classes."
                  )

    def __str__(self):
        return self._str('logistic')

    @property
    def coefficients(self):
        '''The fit model coefficients for each lambda.
 
          The dimensions of this array depend on whether of not a two class
        response was used (i.e. was logistic regression performed):

          * If so: A _n_comp_coef * _out_n_lambdas array containing the fit 
            model coefficients for each value of lambda.

          * If not: A _n_comp_coef * n_classes * _out_n_lambdas array containing
            the model coefficients for each level of the response, for each
            lambda.
        '''
        if self.logistic:
            return self._logistic_coef()
        else:
            raise NotImplementedError("Only two class regression is currently "
                                      "implemented."
                  )

    def _logistic_coef(self):
        ccsq = np.squeeze(self._comp_coef)
        return ccsq[:np.max(self._n_comp_coef),
                    :self._out_n_lambdas
                ]

    def _max_lambda(self, X, y, weights=None):
        '''Return the maximum value of lambda useful for fitting, i.e. that
        which first forces all coefficients to zero.

          This calculation is derived from the discussion in "Regularization 
        Paths for Generalized Linear Models via Coordinate Descent" in the 
        section "Regularized Logistic Regression", using an analogy with the 
        linear case with weighted samples.  We take the initial coefficients 
        to be:
            \beta_0 = log(p/(1-p))
            \beta_i = 0
        I.e., our initial model is the intercept only model.
        
           There is one complication: the working weights in the quadratic 
        approximation to the logistic loss are not normalized, while the 
        discussion in the linear case makes this assumption.  To compensate
        for this, we must adjust the value of lambda to live on the same
        scale as the weights, which causes the normalization factor to drop
        out of the equation.
        '''
        if issparse(X):
            return self._max_lambda_sparse(X, y, weights)
        else:
            return self._max_lambda_dense(X, y, weights)

    def _max_lambda_dense(self, X, y, weights=None):
        if weights is not None:
            raise ValueError("LogisticNet cannot be fit with weights.")
        X_scaled = preprocessing.scale(X)
        # Working response
        p = np.mean(y)
        working_resp = np.log(p/(1-p)) + (y - p) / (p*(1 - p))
        # Working weights
        working_weights = p*(1 - p) / (X.shape[0])
        # Now mimic the calculation for the quadratic case
        y_wtd = working_resp * working_weights
        dots = y_wtd.dot(X_scaled)
        normfac = np.sum(working_weights)
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / alpha

    def _max_lambda_sparse(self, X, y, weights=None):
        if weights is not None:
            raise ValueError("LogisticNet cannot be fit with weights.")
        # Sparse dot product
        dot = self._get_dot(X)
        # Calculation is modeled on weighted least squares
        p = np.mean(y)
        working_resp = np.log(p/(1-p)) + (y - p) / (p*(1 - p))
        working_weights = p*(1 - p) / (X.shape[0])
        # Sample expectataion value of the columns in a matrix
        E = lambda M: np.asarray(M.sum(axis=0)).ravel() / M.shape[0]
        mu = E(X)
        mu_2 = E(X.multiply(X))
        sigma = np.sqrt(mu_2 - mu*mu)
        # Calculating the dot product of y with X standardized, without 
        # destorying the sparsity of X
        y_wtd = working_resp * working_weights
        dots = 1/sigma * (dot(y_wtd, X) - mu * np.sum(y_wtd))
        normfac = np.sum(working_weights)
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / alpha

    def predict(self, X):
        '''Return model predictions on the probability scale.'''
        return 1 / ( 1 + np.exp(self._predict_lp(X)) )

    def deviance(self, X, y):
        '''Calculate the binomial deviance for every lambda.'''
        y_hat = self.predict(X)
        # Take the response y, and repeat it as a column to produce a matrix
        # of the same dimensions as y_hat
        y_stacked = np.tile(np.array([y]).transpose(), y_hat.shape[1])
        #print y_stacked
        bin_dev = y_stacked*np.log(y_hat) + (1 - y_stacked)*np.log(1 - y_hat)
        normfac = X.shape[0]
        return np.apply_along_axis(np.sum, 0, -2*bin_dev) / normfac

    def plot_paths(self):
        self._plot_path('logistic')
