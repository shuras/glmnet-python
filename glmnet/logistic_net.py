import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
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

    def __init__(self, max_iterations=10, opt_type=1, **kwargs):
        '''LogNet accepts the following two configuration parameters on
        configuration.

          * max_iterations: The maximum number of descent iterations to perform
            for each value of lambda.
          * opt_type: The optimization pocedure used to fit each model.
              0: Newton-Ralphson
              1: Modified Newrons.  This is recommended in the fortran
              documentation.
        '''
        GlmNet.__init__(self, **kwargs)
        # The maximum number of iterations to preform for a single lambda
        self._max_iterations = np.array([max_iterations])
        # The optimizations type to perform
        self._opt_type = np.array([opt_type])

    def _fit(self, X, y):
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

          * intecepts: An array of langth _out_n_labdas.  The intercept for
            each model.
          * r_sqs: An array of length _out_n_lambdas containing the r-squared
            statistic for each model.
          * out_lambdas: An array containing the lambda values associated with
            each fit model.
        '''
        X = np.asanyarray(X)
        # Fortran expects an n_obs * n_classes array.  If a one dimensional
        # array is passed, we construct an appropriate widening. 
        y = np.asanyarray(y)
        if len(y.shape) == 1:
            self.logistic = True
            y_classes = np.unique(y)
            y = np.float64(np.column_stack(y == c for c in y_classes))
        else:
            self.logistic = False
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # The target array will usually be overwritten with its standardized
        # version, if this is not ok, we should copy.
        if np.isfortran(X) and not self.overwrite_targ_ok:
            y = y.copy(order='F')
        # Count the number of input levels of y.
        y_level_count = np.unique(y).shape[0]
        # Two class predictions are handled as a special case, as is usual 
        # with logistic models
        if y_level_count == 2:
            self.y_level_count = np.array([1])
        else:
            self.y_level_count = np.array([y_level_count])
        # Setup is complete, call the wrapper
        (self._out_n_lambdas,
        self._intercepts,
        self._comp_coef,
        self._p_comp_coef,
        self._n_comp_coef,
        self.exp_dev,
        self.out_lambdas,
        self._n_passes,
        self._error_flag) = _glmnet.lognet(self.alpha, 
                                           self.y_level_count,
                                           X, 
                                           y, 
                                           self.excl_preds, 
                                           self.rel_penalties,
                                           self.max_vars_all, 
                                           self.frac_lg_lambda, 
                                           self.lambdas,
                                           self.threshold, 
                                           nlam=self.n_lambdas
                            )
        # Keep some model metadata
        self._n_fit_obs, self._n_fit_params = X.shape
        # The indexes into the predictor array are off by one due to fortran
        # convention, fix it up.
        self._indicies = np.trim_zeros(self._p_comp_coef, 'b') - 1
        # Check for errors, documented in glmnet.f.
        if self._error_flag != 0:
            if self._error_flag == 10000:
                raise ValueError('cannot have max(vp) < 0.0')
            elif self._error_flag == 7777:
                raise ValueError('all used predictors have 0 variance')
            elif self._error_flag < 7777:
                raise MemoryError('elnet() returned error code %d' % jerr)
            else:
                raise Exception('unknown error: %d' % jerr)

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
        return np.apply_along_axis(np.sum, 0, -2*bin_dev)

    def _plot_path(self):
        self._plot_path('logistic')
