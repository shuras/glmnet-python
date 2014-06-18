import numpy as np
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed

def _fit_and_score(glmnet, X, y, train_inds, test_inds):
    glmnet.fit(X[train_inds], y[train_inds])
    return (glmnet.out_lambdas[:glmnet._out_n_lambdas], 
            glmnet.deviance(X[test_inds], y[test_inds])
           )

def _clone(glmnet):
    return glmnet.__class__(**glmnet.__dict__)

class CVGlmNet(object):
    '''Determine the optimal lambda parameter for a glmnet by cross 
    validation.
    '''

    def __init__(self, glmnet, 
                 n_folds=3, n_jobs=3, shuffle=True, verbose=2
        ):
        '''Create a cross validation glmnet object.  Accepts the following
        arguments:

          * glmnet: An object derived from the GlmNet class, currently either
              an ElasticNet or LogisticNet object.
          * n_folds: The number of cross validation folds.
          * n_jobs: The number of cores to distribute the work to.
          * shuffle: Boolean, should the indicies of the cross validation 
              folds be shuffled randomly.
          * verbose: Ammount of talkyness.
        '''
        self._base_estimator = glmnet
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.verbose = verbose

    def fit(self, X, y):
        '''Determine the optimal value of lambda by cross validation, and fit
        a single glmnet with this value of lambda.
        '''
        # Determine the indicies of the various train and test sets for the 
        # cross validation.
        cv_folds = KFold(X.shape[0], n_folds=self.n_folds, shuffle=self.shuffle) 
        # Copy the glmnet to fit as the final model.
        base_estimator = _clone(self._base_estimator)
        # Fit in-fold glmnets in parallel.  For each such model, pass back the 
        # series of lambdas fit and the out-of-fold deviances for each such 
        # lambda.
        ld_pairs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                     delayed(_fit_and_score)(_clone(base_estimator), X, y,
                                               train_inds, test_inds
                                            )
                     for train_inds, test_inds in cv_folds
                   )
        self.lambdas = tuple(ld[0] for ld in ld_pairs)
        self.deviances = tuple(ld[1] for ld in ld_pairs)
        # Determine the best lambda in each series, i.e. the value of lambda
        # that minimizes the out of fold deviance.
        best_lambdas = tuple(lambdas[np.argmin(devs)] 
                             for lambdas, devs in ld_pairs
                       )
        # The optimal lambda
        self.best_lambda = np.mean(best_lambdas)
        # Refit a glmnet on the entire data set uning the value of lambda
        # determined to be optimal
        base_estimator.lambdas = [self.best_lambda]
        base_estimator.fit(X, y)
        self.best_model = base_estimator

    @property
    def intercepts(self):
        return self.best_model.intercepts.squeeze()

    @property
    def coefficients(self):
        return self.best_model.coefficients.squeeze()

    def predict(self, X):
        return self.best_model.predict(X).squeeze()
