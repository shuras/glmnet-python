import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
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

    def _fit(self, X, y):
        '''Fit an elastic net model.

        Arguments: 

          * X: The predictors.  A n_obs * n_preds array.
          * y: The response.  A n_obs array.

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
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # The target array will usually be overwritten with its standardized
        # version, if this is not ok, we should copy.
        if not self.overwrite_targ_ok:
            y = y.copy()
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
                                          self.max_vars_all, 
                                          self.frac_lg_lambda, 
                                          self.lambdas, 
                                          self.thresh, 
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
        s = ("An elastic net model fit on %d observations and %d parameters.\n"
             "The model was fit in %d passes over the data.                 \n"
             "There were %d values of lambda resulting in non-zero models.  \n"
             "There were %d non-zero coefficients in the largest model.     \n")
        return s % (self._n_fit_obs, self._n_fit_params,
                        self._n_passes,
                        self._out_n_lambdas,
                        np.max(self._n_comp_coef)
               )

    @property
    def coefficients(self):
        '''The fit model coefficients for each lambda.

          A _n_comp_coef * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        return self._comp_coef[:np.max(self._n_comp_coef),
                               :self._out_n_lambdas
                ]

    @property
    def intercepts(self):
        '''The fit model intercepts.

          A _n_comp_coef * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        return self._intercepts[:self._out_n_lambdas]

    def predict(self, X):
        '''Produce model predictions from new data.'''
        return self.intercepts + np.dot(X[:, self._indicies],
                                        self.coefficients
                                 )

    def _plot_path(self):
        '''Plot the full regularization path of all the non-zero model
        coefficients.
        '''
        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.out_lambdas[1:self._out_n_lambdas])
        for coef_path in self.coefficients:
            ax.plot(xvals, coef_path[1:])
        ax.set_title("Regularization paths for elastic net with alpha = %s" % 
                     self.alpha)
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Parameter Value")
        plt.show()
    
