import numpy as np
from sklearn import preprocessing
import _glmnet
from glmnet import GlmNet

class ElasticNet(GlmNet):
    '''The elastic net: a multivariate linear model with both L1 and L2
    regularizers.

      This class implements the elastic net class of predictive models. These
    models combine the classical ridge and lasso regularizers into a combined
    penalty.  More specifically, the loss function minimized by this model is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            RSS(\beta_0, \beta_1, ..., \beta_n; X, y) + 
            \lambda * ( (\alpha - 1)/2 * | \beta |_2 + \alpha * | \beta |_1 )

    where RSS is the usual residual sum of squares:

      RSS(\beta_0, \beta_1, ..., \beta_n; X, y) = sum( (\beta * X_i - y_i)^2 )
    '''

    def fit(self, X, y,
            lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None):
        '''Fit an elastic net model.

        Arguments: 

          * X: The predictors.  A n_obs * n_preds array.
          * y: The response.  A n_obs array.

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

        After fitting, the following attributes are set:
        
        Private attributes:

          * _out_n_lambdas: The number of fit lambdas associated with non-zero
            models; for large enough lambdas the models will become zero in the
            presense of an L1 regularizer.
          * _intecepts: An array of langth _out_n_labdas.  The intercept for
            each model.
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

          * r_sqs: An array of length _out_n_lambdas containing the r-squared
            statistic for each model.
          * out_lambdas: An array containing the lambda values associated with
            each fit model.
        '''
        # Predictors and response
        try:
            X = np.asanyarray(X)
            y = np.asanyarray(y)
        except ValueError:
            raise ValueError("X and y must be wither numpy arrays, or "
                             "convertable to numpy arrays."
                  )
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # The target array will usually be overwritten with its standardized
        # version, if this is not ok, we should copy.
        if not self.overwrite_targ_ok:
            y = y.copy()
        # Validate all the inputs:
        self._validate_inputs(X, y)
        self._validate_lambdas(X, y, lambdas)
        self._validate_weights(X, y, weights)
        self._validate_rel_penalties(X, y, rel_penalties)
        self._validate_excl_preds(X, y, excl_preds)
        self._validate_box_constraints(X, y, box_constraints)
        # Setup is complete, call the wrapper.
        (self._out_n_lambdas,
        self._intercepts,
        self._comp_coef,
        self._p_comp_coef,
        self._n_comp_coef,
        self.r_sqs,
        self.out_lambdas,
        self._n_passes,
        self._error_flag) = _glmnet.elnet(self.alpha, 
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
        self._check_errors()
        # Keep some model metadata
        self._n_fit_obs, self._n_fit_params = X.shape
        # The indexes into the predictor array are off by one due to fortran
        # convention, fix it up.
        self._indicies = np.trim_zeros(self._p_comp_coef, 'b') - 1

    def __str__(self):
        return self._str('elastic')

    @property
    def coefficients(self):
        '''The fit model coefficients for each lambda.

          A _n_comp_coef * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        return self._comp_coef[:np.max(self._n_comp_coef),
                               :self._out_n_lambdas
                ]

    def _max_lambda(self, X, y):
        '''Return the maximum value of lambda useful for fitting, i.e. that
        which first forces all coefficients to zero.
        '''
        X_scaled = preprocessing.scale(X)
        dots = y.dot(X_scaled)
        return np.max(np.abs(dots)) / (self.alpha * X.shape[0])

    def predict(self, X):
        '''Produce model predictions from new data.'''
        return self._predict_lp(X)

    def deviance(self, X, y, weights = None):
        '''Calculate the normal deviance (i.e. sum of squared errors) for
        every lambda.'''
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
        else:
            w_stacked = np.tile(np.array([weights]).transpose(),
                                y_hat.shape[1]
                        )
            sq_residuals = w_stacked * (y_stacked - y_hat)**2
        # Determine the appropriate normalization factor:
        if weights is None:
            normfac = X.shape[0]
        else:
            normfac = np.sum(weights)
        return np.apply_along_axis(np.sum, 0, sq_residuals) / normfac

    def plot_path(self):
        self._plot_path('elastic') 
