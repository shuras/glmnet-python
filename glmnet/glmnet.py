import numpy as np
from scipy.sparse import issparse, isspmatrix_csc
import matplotlib
import matplotlib.pyplot as plt 
from glmnet_config import (_DEFAULT_THRESH,
                           _DEFAULT_FLMIN,
                           _DEFAULT_NLAM)
from warnings import warn

class GlmNet(object):
    '''Parent class for glmnet model objects.
    
      Glmnets are a class of predictive models. They are a regularized version
    generalized linear models that combines the ridge (L^2) and lasso (L^1)
    penalties.  The general form of the loss function being optimized is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            Dev(\beta_0, \beta_1, ..., \beta_n) + 
            \lambda * ( (\alpha - 1)/2 * | \beta |_2 + \alpha * | \beta |_1 )

    where Dev is the deviance of the classical glm, |x|_2 and |x|_1 are the L^2
    and L^1 norms, and \lambda and \alpha are tuning parameters:

      * \lambda controlls the overall ammount of regularization, and is usually
        tuned by cross validation.

      * \alpha controlls the balance between the L^1 and L^2 regularizers. In
        the extreme cases: \alpha = 0 : Ridge Regression
                           \alpha = 0 : Lasso Regression

    All glmnet object accept a value of \alpha at instantiation time.  Glmnet
    defaults to fitting a full path of \lambda values, from \lambda_max (all
    parameters zero) to 0 (an unregularized model).  The user may also choose to
    supply a list of alphas, in this case the default behavior is overriden and
    a glmnet is fit for each value of lambda the user supplies.
    '''

    def __init__(self, 
                 alpha, 
                 offsets=None,
                 standardize=True,
                 max_vars_all=None,
                 max_vars_largest=None,
                 threshold=_DEFAULT_THRESH,
                 frac_lg_lambda=_DEFAULT_FLMIN,
                 n_lambdas=_DEFAULT_NLAM,
                 overwrite_pred_ok=False,
                 overwrite_targ_ok=False
        ):
        '''Create a glmnet object and implement configuration common to all
        subclasses.  Accepts the following arguments:

          * alpha: Relative weighting between the L1 and L2 regularizers. 
            parameter.
          * standardize: Boolean flag, do we standardize the predictor
            variables.  Defaults to true, which is important for the regularizer
            to be fair.  Note that the output parameters are allways reported on
            the scale of the origional predictors.
          * max_vars_all: Bound to inforce on the number of variables in all
            models.
          * max_vars_largest: Bound on the number of variables that enter in the
            largest model.
          * threshold: Convergence threshold for each lambda.
          * frac_lg_lambda: Control parameter for range of lambda values to
          * search: \lambda_min = frac_lg_lambda *  (\lambda_max)   
          * n_lambdas: The number of lambdas to fit in the search space.
          * overwrite_pred_ok: Overwirte the memory holding the predictor when
            standardizing.
          * overwirte_targ_ok: Overwrite the memory holding the target when
            standardizing.
        '''
        # Relative weighting between L1 and L2 norm
        self.alpha = alpha
        # Standardize input variables?
        self.standardize = standardize
        # The maximum number of parameters allowed to be nonzero in any model
        self.max_vars_all = max_vars_all
        # The maximum number of parameters allowed to be nonzero in the
        # largest model
        self.max_vars_largest = max_vars_largest
        # Minimum change in largest coefficient, stopping criterion
        self.threshold = threshold 
        # Fraction of largest lambda at which to stop
        self.frac_lg_lambda = _DEFAULT_FLMIN
        # Maximum number of lambdas to try
        self.n_lambdas = _DEFAULT_NLAM
        # Not sure about these right now...
        self.overwrite_pred_ok = overwrite_pred_ok
        self.overwrite_targ_ok = overwrite_targ_ok 

    def _validate_lambdas(self, X, y, lambdas):
        '''glmnet functions expect either a user supplied array of lambdas, or
        a signal to construct its own.
        '''
        if lambdas is not None:
            self.lambdas = np.asarray(lambdas)
            self.n_lambdas = len(lambdas)
            # Pass >1 to indicate that the user passed in a list of lambdas
            self.frac_lg_lambda = 2
        else:
            self.lambdas = None

    def _validate_weights(self, X, y, weights):
        '''If no explicit weights are passed, each observation is given the 
        same weight
        '''
        self.weights = (np.ones(X.shape[0]) if weights is None
                                            else weights
                       )
        if self.weights.shape[0] != X.shape[0]:
            raise ValueError("The weights vector must have the same length "
                             "as X."
                  )

    def _validate_rel_penalties(self, X, y, rel_penalties):
        '''If no explicit penalties are passed, each varaible is given the same
        penalty
        '''
        self.rel_penalties = (np.ones(X.shape[1]) if rel_penalties is None
                                                  else rel_penalties
                             )
        if self.rel_penalties.shape[0] != X.shape[1]:
            raise ValueError("The relative penalties vector must have the "
                             "same length as the number of columns in X."
                  )

    def _validate_excl_preds(self, X, y, excl_preds):
        '''If no explicit exclusion is supplied, pass a zero to exclude 
        nothing.
        '''
        self.excl_preds = (np.zeros(1) if excl_preds is None
                                       else excl_preds
                          )
        if self.excl_preds.shape[0] != 1:
            if self.excl_preds.shape[0] != X.shape[1]:
                raise ValueError("Non null excluded predictors array must "
                                 "have the same length as the number of "
                                 "columns in X."
                      )

    def _validate_box_constraints(self, X, y, box_constraints):
        '''Box constraints on parameter estimates.'''
        if box_constraints is None:
            bc = np.empty((2, X.shape[1]), order='F')
            bc[0,:] = float(-10000)
            bc[1,:] = float(10000)
            self.box_constraints = bc
        elif box_constraints.shape[1] != X.shape[1]:
            raise ValueError("Box constraints must be a vector of shape 2, "
                            "number of columns in X."
                  )
        self.box_constraints = self.box_constraints.copy(order='F')

    def _validate_inputs(self, X, y):
        '''Validate and process the prectors and response for model fitting.'''
        # Check that the dimensions work out
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length.")
        # Decide on the largest allowable models
        self.max_vars_all = (
            X.shape[1] if self.max_vars_all is None else self.max_vars_all
        )
        self.max_vars_largest = (
            self.max_vars_all if self.max_vars_largest is None
                              else self.max_vars_largest
        )
        if self.max_vars_all < self.max_vars_largest:
            raise ValueError("Inconsistant parameters: need max_vars_all "
                             "< max_vars_largest."
                  )

    @staticmethod
    def _validate_matrix(X):
        if issparse(X) and not isspmatrix_csc(X):
            raise ValueError("Sparse matrix detected, but not in compressed "
                             "sparse row format."
                  )

    @staticmethod
    def _get_dot(X):
        if issparse(X):
            return X.dot.__func__
        else:
            return np.dot

    def _check_errors(self):
        '''Check for errors, documented in glmnet.f.'''
        # Fatal errors...
        if self._error_flag > 0:
            if self._error_flag == 10000:
                raise ValueError('Fatal: Cannot have max(vp) < 0.0.')
            elif self._error_flag == 7777:
                raise ValueError('Fatal: all used predictors have 0 variance.')
            elif self._error_flag < 7777:
                raise MemoryError('Fatal: Memory allocation error.')
            else:
                raise Exception('Fatal: Unknown error code: %d' 
                                % self._error_flag
                      )
        # Non-fatal errors...
        elif self._error_flag < 0:
            if self._error_flag > -10000:
                last_lambda = -self._error_flag
                w_msg = ("Convergence for {0:d}'th lambda value not reached "
                         "after {1:d} iterations.")
                warn(w_msg.format(last_lambda, self._n_passes), RuntimeWarning)
            elif self._error_flag <= -10000:
                last_lambda = -(self._error_flag + 10000)
                w_msg = ("Number of non-zero coefficients exceeds {0:d} at "
                         "{1:d}th lambda value.")
                warn(w_msg.format(self.max_vars_all, last_lambda),
                     RuntimeWarning
                )
            else:
                warn("Unknown warning %d" % self._error_flag)

        
    def _str(self, name):
        s = ("A %s net model fit on %d observations and %d parameters.     \n"
             "The model was fit in %d passes over the data.                \n"
             "There were %d values of lambda resulting in non-zero models. \n"
             "There were %d non-zero coefficients in the largest model.    \n")
        return s % (name, 
                    self._n_fit_obs, self._n_fit_params,
                    self._n_passes,
                    self._out_n_lambdas,
                    np.max(self._n_comp_coef)
               )

    def _clone(self):
        '''Copy an unfit glmnet object.'''
        return self.__class__(**self.__dict__)

    @property
    def intercepts(self):
        '''The fit model intercepts.

          A _n_comp_coef * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        return self._intercepts.ravel()[:self._out_n_lambdas]

    def _predict_lp(self, X):
        '''Return model predictions on a linear predictor scale.

          Returns an n_obs * n_lambdas array, where n_obs is the number of rows
        in X.
        '''
        dot = self._get_dot(X)
        return self.intercepts + dot(X[:, self._indicies],
                                        self.coefficients
                                    )

    def _plot_path(self, name):
        '''Plot the full regularization path of all the non-zero model
        coefficients.
        '''
        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.out_lambdas[1:self._out_n_lambdas])
        for coef_path in self.coefficients:
            ax.plot(xvals, coef_path[1:])
        ax.set_title("Regularization paths for %s net with alpha = %s" % 
                     (name, self.alpha))
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Parameter Value")
        plt.show()
