import numpy as np
from scipy.sparse import issparse, isspmatrix_csc
from glmnet_config import (_DEFAULT_THRESH,
                           _DEFAULT_FLMIN,
                           _DEFAULT_NLAM)
from warnings import warn

from ..util.importers import import_pyplot
plt = import_pyplot()


class GlmNet(object):
    '''Parent class for glmnet model objects.

    Glmnet Info:
    
      Glmnets are a class of predictive models. They are a regularized version
    of generalized linear models that combines the ridge (L^2) and lasso (L^1)
    penalties.  The general form of the loss function being optimized is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            Dev(\beta_0, \beta_1, ..., \beta_n) + 
            \lambda * ( (1 - \alpha)/2 * | \beta |_2 + \alpha * | \beta |_1 )

    where Dev is the deviance of a classical glm, |x|_2 and |x|_1 are the L^2
    and L^1 norms, and \lambda and \alpha are tuning parameters:

      * \lambda controlls the overall ammount of regularization, and is usually
        tuned by cross validation.

      * \alpha controlls the balance between the L^1 and L^2 regularizers. In
        the extreme cases: \alpha = 0 : Ridge Regression
                           \alpha = 1 : Lasso Regression

    All glmnet objects accept a value of \alpha at instantiation time.  Glmnet
    defaults to fitting a full path of \lambda values, from \lambda_max (all
    parameters zero) to 0 (an unregularized model).  The user may also choose to
    supply a list of \lambdas, in this case the default behavior is overriden and
    a glmnet is fit for each value of lambda the user supplies.

      The function Dev depends on the specific type of glmnet under 
    consideration.  Different choices of Dev determine various predictive
    models in the glmnet family.  For details on the different types of
    glmnets, the reader should consult the various subclasses of GlmNet.

    This Class:

      This class is the parent of all glment models.  As such, it only
    implements functionality in common to all glmnets, independednt of the 
    specific loss function; this includes data needed to instantiate generic
    glmnets, checking inputs for validity, and checking error codes after 
    fitting.
    '''

    def __init__(self, 
                 alpha, 
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

          * alpha: 
              Relative weighting between the L1 and L2 regularizers. 
          * standardize: 
              Boolean flag, do we standardize the predictor variables?  Defaults
              to true, which is important for the regularizer to be fair.  Note
              that the output parameters are allways reported on the scale of
              the origional predictors.
          * max_vars_largest: 
              Maximum number of variables allowed in the largest model.  This 
              acts as a stopping criterion.
          * max_vars_all: 
              Maximum number of non-zero variables allowed in any model.  This 
              controls memory alocation inside glmnet.
          * threshold: 
              Convergence threshold for each lambda.  For each lambda,
              iteration is stopped when imporvement is less than threashold.
          * frac_lg_lambda: 
              Control parameter for range of lambda values to search:
                \lambda_min = frac_lg_lambda *  (\lambda_max)   
              where \lambda_max is calcualted based on the data and the 
              model type.
          * n_lambdas: 
              The number of lambdas to include in the grid search.
          * overwrite_pred_ok: 
              Boolean, overwirte the memory holding the predictor when 
              standardizing?
          * overwirte_targ_ok: 
              Boolean, overwrite the memory holding the target when
              standardizing?
        '''
        self.alpha = alpha
        self.standardize = standardize
        self.max_vars_all = max_vars_all
        self.max_vars_largest = max_vars_largest
        self.threshold = threshold 
        self.frac_lg_lambda = _DEFAULT_FLMIN
        self.n_lambdas = _DEFAULT_NLAM
        self.overwrite_pred_ok = overwrite_pred_ok
        self.overwrite_targ_ok = overwrite_targ_ok 

    def _validate_lambdas(self, X, y, lambdas):
        '''glmnet expects either a user supplied array of lambdas, or a signal
        to construct its own.
        '''
        if lambdas is not None:
            self.lambdas = np.asarray(lambdas)
            self.n_lambdas = len(lambdas)
            # >1 indicates that the user passed in a list of lambdas
            self.frac_lg_lambda = 2
        else:
            self.lambdas = None

    def _validate_weights(self, X, y, weights):
        '''If no explicit sample weights are passed, each observation is given a
        unit weight.
        '''
        self.weights = (np.ones(X.shape[0]) if weights is None
                                            else weights
                       )
        if self.weights.shape[0] != X.shape[0]:
            raise ValueError("The weights vector must have the same length "
                             "as X."
                  )
        if np.any(self.weights < 0):
            raise ValueError("All sample weights must be non-negative.")

    def _validate_rel_penalties(self, X, y, rel_penalties):
        '''If no explicit penalty weights are passed, each varaible is given the
        same penalty weight.
        '''
        self.rel_penalties = (np.ones(X.shape[1]) if rel_penalties is None
                                                  else rel_penalties
                             )
        if self.rel_penalties.shape[0] != X.shape[1]:
            raise ValueError("The relative penalties vector must have the "
                             "same length as the number of columns in X."
                  )
        if np.any(self.rel_penalties < 0):
            raise ValueError("All relative penalties must be non-negative.")
        if np.max(self.rel_penalties) <= 0:
            raise ValueError("Must have at least one positive relative "
                             "penalty."
                  )

    def _validate_excl_preds(self, X, y, excl_preds):
        '''If no explicit exclusion is supplied, pass a zero to exclude nothing.
        '''
        self.excl_preds = (np.zeros(1) if excl_preds is None
                                       else excl_preds
                          )
        if self.excl_preds.shape[0] != 1:
            if self.excl_preds.shape[0] > X.shape[1] + 1:
                raise ValueError("Non null excluded predictors array must "
                                 "have less entries than the number of "
                                 "columns in X."
                      )
            if np.any(self.excl_preds[1:] >= X.shape[1]):
                raise ValueError("Entries in non null excluded predictors "
                                 "array (except for the first entry) must "
                                 "enumerate columns in X."
                      )

    def _validate_box_constraints(self, X, y, box_constraints):
        '''Box constraints on parameter estimates.'''
        if box_constraints is None:
            bc = np.empty((2, X.shape[1]), order='F')
            bc[0,:] = float("-inf")
            bc[1,:] = float("inf")
            box_constraints = bc
        if (box_constraints.shape[1] != X.shape[1] or 
              box_constraints.shape[0] != 2):
            raise ValueError("Box constraints must be a vector of shape 2, "
                            "number of columns in X."
                  )
        if (np.any(box_constraints[0,:] > 0) or
            np.any(box_constraints[1,:] < 0)):
            raise ValueError("Box constraints must be intervals of the form "
                             "[non-positive, non-negative] for each "
                             "predictor."
                  )
        self.box_constraints = box_constraints.copy(order='F')

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
            raise ValueError("Inconsistant parameters: need max_vars_largest "
                             "< max_vars_all."
                  )

    @staticmethod
    def _validate_matrix(X):
        '''glmnet only accepts sparse matricies in compressed saprse column 
        format.  Note: while glment documentation says it wants sparse row
        format, it lies.
        '''
        if issparse(X) and not isspmatrix_csc(X):
            raise ValueError("Sparse matrix detected, but not in compressed "
                             "sparse row format."
                  )


    def _check_errors(self):
        '''Check for errors, documented in glmnet.f.'''
        # Fatal errors.
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
        # Non-fatal errors.
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

    @property
    def intercepts(self):
        '''The fit model intercepts, one for each value of lambda.'''
        self._check_if_fit()
        return self._intercepts.ravel()[:self._out_n_lambdas]

    def get_coefficients_from_lambda_idx(self, lidx):
        '''Get the array of coefficient estimates given an index into the
        array of fit lambdas, that is, for a fixed lambda (index) return the
        estimates of the various coefficients given lambda.

          An array of length _n_fit_params (i.e. X.shape[1]).
        '''
        coefs = np.zeros(shape=(self._n_fit_params,))
        coefs[self._indices] = self._coefficients[:,lidx]
        return coefs

    def get_coefficients_from_col_idx(self, cidx):
        '''Get the array of coefficient estimates given the index of a 
        column in the model matrix, that is, for a fixed column return all the
        various estimates of the associated coefficient across lambda.  Returns
        the zero array if the coefficient is compressed out of _coefficients.

          An array of length _out_n_lambdas.
        '''
        if cidx not in self._indices:
            return np.zeros(shape=(self._out_n_lambdas,))
        else:
           ridx = np.where(self._indices == cidx)
           return self._coefficients[ridx,:].squeeze()

    def _describe(self, lidx=None, name='glmnet'):
        '''Display information about the model.  Behaviour depends on 
        state of the model when called, and whether a specific lambda index
        is passed.
        '''
        display_str = self._str(name)
        if self._is_fit():
            display_str = display_str + self._fit_str(name)
        if lidx != None:
            sep = '-'*79 + '\n'
            display_str = display_str + sep + self._coef_str(lidx, name)
        return display_str.strip()

    def _str(self, name):
        '''A generic message pertaining to all glmnets.'''
        return 'A {name} net model with alpha = {alpha}.\n'.format(
                   name=name, alpha=self.alpha
               )

    def _fit_str(self, name):
        '''A generic message pertaining to all fit glmnets.'''
        self._check_if_fit()
        s = ("The model was fit on {0} observations and {1} parameters.     \n"
             "The model was fit in {2} passes over the data.                \n"
             "There were {3} values of lambda resulting in non-zero models. \n"
             "There were {4} non-zero coefficients in the largest model.    \n")
        return s.format(self._n_fit_obs, self._n_fit_params,
                        self._n_passes,
                        self._out_n_lambdas,
                        np.max(self._n_comp_coef)
               )

    def _coef_str(self, lidx, name):
        '''Create a string containing a table of coefficient estimates for
        a given value of lambda.
        '''
        lam = self.out_lambdas[lidx]
        coefs = self.get_coefficients_from_lambda_idx(lidx)
        title = ("Parameter Estiamtes for {net} net model with "
                 "lambda = {lam:1.5e}\n").format(net=name, lam=lam)
        header = "{nameh:<40}{valh:<10}\n".format(
            nameh="Varaible Name", valh="Coefficent Estiamte"
        )
        line_template = "{name:<40}{val:1.5e}"
        body = "\n".join(
            line_template.format(name=self._col_names[i], val=coefs[i])
            for i in range(self._n_fit_params)
        )
        return title + header + body
        
    def _predict_lp(self, X):
        '''Model predictions on a linear predictor scale.

          Returns an n_obs * n_lambdas array, where n_obs is the number of rows
        in X.
        '''
        self._check_if_fit()
        dot = self._get_dot(X)
        if np.max(self._n_comp_coef) > 0 :
            return self.intercepts + dot(X[:, self._indices],
                                        self._coefficients
                                        )
        else :
            return np.tile(self.intercepts, (X.shape[0], 1))

    def _plot_path(self, name):
        '''Plot the full regularization path of all the non-zero model
        coefficients.  Creates an displays a plot of the parameter estimates
        at each value of log(\lambda).
        '''
        self._check_if_fit()
        if not plt:
            raise RuntimeError('pyplot unavailable.')

        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.out_lambdas[1:self._out_n_lambdas])
        for coef_path in self._coefficients:
            ax.plot(xvals, coef_path[1:])
        ax.set_title("Regularization paths for %s net with alpha = %s" % 
                     (name, self.alpha))
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Parameter Value")
        plt.show()

    def _is_fit(self):
        '''The model has been fit successfully if and only if the _n_fit_obs
        attribute exists.
        '''
        return hasattr(self, '_n_fit_obs')

    def _check_if_fit(self, reverse=False):
        '''Raise exception if model is not fit.'''
        its_ok = (not self._is_fit()) if reverse else self._is_fit()
        word = 'already' if reverse else 'not'
        if its_ok:
            return
        else:
            raise RuntimeError('The operation cannot be performed on a model '
                               'that has ' + word + ' been fit.'
                  )

    def _check_if_unfit(self):
        return self._check_if_fit(reverse=True)

    @staticmethod
    def _get_dot(X):
        '''Get the underlying function for a dot product of two matricies, 
        independent of type.  This allows us to write dot(X, Y) for both 
        dense and sparse matricies.
        '''
        if issparse(X):
            return X.dot.__func__
        else:
            return np.dot
