import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import _glmnet
from glmnet import GlmNet

class ElasticNet(GlmNet):

    def _fit(self, X, y):
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

        # Setup is complete, call the wrapper
        (self._out_n_lambdas,
        self.intercepts,
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
        return self._comp_coef[:np.max(self._n_comp_coef),
                               :self._out_n_lambdas
                ]

    def predict(self, X):
        return self.intercepts + np.dot(X[:, self._indicies],
                                        self.coefficients
                                 )

    def _plot_path(self):
        plt.clf()
        xvals = np.log(self.out_lambdas[1:self._out_n_lambdas])
        for coef_path in self.coefficients:
            plt.plot(xvals, coef_path[1:])
        plt.show()
    
