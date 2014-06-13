import numpy as np
from glmnet_config import (_DEFAULT_THRESH,
                           _DEFAULT_FLMIN,
                           _DEFAULT_NLAM)

class GlmNet(object):

    def __init__(self, 
                 alpha, 
                 lambdas=None,
                 weights=None,
                 rel_penalties=None,
                 excl_preds=None,
                 standardize=True,
                 max_vars_all=None,
                 max_vars_largest=None,
                 threshold=_DEFAULT_THRESH,
                 frac_lg_lambda=_DEFAULT_FLMIN,
                 n_lambdas=_DEFAULT_NLAM,
                 overwrite_pred_ok=False,
                 overwrite_targ_ok=False
        ):
        '''Configure the glmnet.'''
        # Relative weighting between L1 and L2 norm
        self.alpha = alpha
        # User supplied lambdas
        self.lambdas = lambdas
        # Weighting for each predictor
        self.weights = weights
        # Relative penalties for each predictor varaibles, 0 is unpenalized
        self.rel_penalties = rel_penalties
        # Predictors to exclude from all models
        self.excl_preds = excl_preds
        # Standardize input variables?
        self.standardize = standardize
        # The maximum number of parameters allowed to be nonzero in any model
        self.max_vars_all = max_vars_all
        # The maximum number of parameters allowed to be nonzero in the
        # largest model
        self.max_vars_largest = max_vars_largest
        # Minimum change in largest coefficient, stopping criterion
        self.thresh = threshold 
        # Fraction of largest lambda at which to stop
        self.frac_lg_lambda = _DEFAULT_FLMIN
        # Maximum number of lambdas to try
        self.n_lambdas = _DEFAULT_NLAM
        # Not sure about these right now...
        self.overwrite_pred_ok = overwrite_pred_ok
        self.overwrite_targ_ok = overwrite_targ_ok 

    def _validate_inputs(self, X, y):

        X = np.asanyarray(X)
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
                             "< max_vars_larges."
                  )

        # If no explicit weights are passed, each observation is given the same
        # weight
        self.weights = (np.ones(X.shape[0]) if self.weights is None
                                            else self.weights
                       )
        # If no explicit penalties are passed, each varaible is given the same
        # penalty
        self.rel_penalties = (np.ones(X.shape[1]) if self.rel_penalties is None
                                                  else self.rel_penalties
                             )
        # If no explicit exclusion is supplied, pass a zero to exclude nothing
        self.excl_preds = (np.zeros(1) if self.excl_preds is None
                                       else self.excl_preds
                          )
        # User supplied list of lambdas
        if self.lambdas is not None:
            self.lambdas = np.asarray(self.lambdas)
            self.n_lambdas = len(self.lambdas)
            # Pass >1 to indicate that the user passed in a list of lambdas
            self.frac_lg_lambda = 2
        else:
            self.lambdas = None
        
    def fit(self, X, y):
        self._validate_inputs(X, y)
        self._fit(X, y)

    def plot_path(self):
        self._plot_path()
