import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import _glmnet
from glmnet import GlmNet

class LogNet(GlmNet):

    def __init__(self, max_iterations=10, opt_type=1, **kwargs):
        GlmNet.__init__(self, **kwargs)
        # The maximum number of iterations to preform for a single lambda
        self._max_iterations = np.array([max_iterations])
        # The optimizations type to perform
        self._opt_type = np.array([opt_type])

    def _fit(self, X, y):
        # Predictors and response
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
        self.intercepts,
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
                                           self.thresh, 
                                           nlam=self.n_lambdas
                            )
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

    @property
    def coefficients(self):
        if self.logistic:
            return self._logistic_coef()
        else:
            return self._non_logistic_coef()

    def _logistic_coef(self):
        ccsq = np.squeeze(self._comp_coef)
        return ccsq[:np.max(self._n_comp_coef),
                    :self._out_n_lambdas
                ]

    def _predict_lp(self, X):
        return np.dot(X[:, self._indicies],
                      self.coefficients
        )

    def predict(self, X):
        return 1 / ( 1 + np.exp(self.intercepts + self._predict_lp(X) ) )

