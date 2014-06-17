from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed

def _fit_and_score(glmnet, X, y, train_inds, test_inds):
    glmnet.fit(X[train_inds], y[train_inds])
    return (glmnet.out_lambdas, glmnet.deviance(X[test_inds], y[test_inds]))

def _clone(glmnet):
    return glmnet.__class__(**glmnet.__dict__)

class CVGlmNet(object):

    def __init__(self, glmnet, 
                 n_folds=3, n_jobs=3, shuffle=True, verbose=2
    ):
        self._base_estimator = glmnet
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.verbose = verbose

    def fit(self, X, y):
        self._validate_inputs(X, y)
        cv_folds = KFold(X.shape[0], n_folds=self.n_folds, shuffle=self.shuffle) 

        base_estimator = _clone(self._base_estimator)

        dl_pairs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                     delayed(_fit_and_score)(_clone(base_estimator), X, y,
                                               train_inds, test_inds
                                            )
                     for train_inds, test_inds in cv_folds
                   )
        print dl_pairs


    def _validate_inputs(self, X, y):
        pass
