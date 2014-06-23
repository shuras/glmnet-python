import numpy as np
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed

def _fit_and_score_elastic_net(elastic_net, X, y, 
                               train_inds, test_inds, 
                               **kwargs):
    if 'weights' in kwargs:
        weights = kwargs['weights']
        train_weights = weights[train_inds]
        test_weights = weights[test_inds]
    else:
        train_weights, test_weights = None, None
    del kwargs['weights']
    elastic_net.fit(X[train_inds], y[train_inds], 
                    weights=train_weights, **kwargs
                )
    return (elastic_net.out_lambdas[:elastic_net._out_n_lambdas], 
            elastic_net.deviance(X[test_inds], y[test_inds], 
                                 weights=test_weights
                        )
           )

def _fit_and_score_logistic_net(logistic_net, X, y, 
                                train_inds, test_inds,
                                **kwargs):
    logistic_net.fit(X[train_inds], y[train_inds], **kwargs)
    return (logistic_net.out_lambdas[:logistic_net._out_n_lambdas], 
            logistic_net.deviance(X[test_inds], y[test_inds])
           )

fit_and_score_switch = {'ElasticNet': _fit_and_score_elastic_net,
                        'LogisticNet': _fit_and_score_logistic_net
                       }

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

    def fit(self, X, y, **kwargs):
        '''Determine the optimal value of lambda by cross validation, and fit
        a single glmnet with this value of lambda.
        '''
        # Determine the indicies of the various train and test sets for the 
        # cross validation.
        cv_folds = KFold(X.shape[0], n_folds=self.n_folds, shuffle=self.shuffle) 
        # Copy the glmnet to fit as the final model.
        base_estimator = _clone(self._base_estimator)
        fit_and_score = fit_and_score_switch[base_estimator.__class__.__name__] 
        # Fit in-fold glmnets in parallel.  For each such model, pass back the 
        # series of lambdas fit and the out-of-fold deviances for each such 
        # lambda.
        ld_pairs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                     delayed(fit_and_score)(_clone(base_estimator), X, y,
                                            train_inds, test_inds,
                                            **kwargs
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
        base_estimator.fit(X, y, lambdas=[self.best_lambda], **kwargs)
        self.best_model = base_estimator

    @property
    def intercepts(self):
        return self.best_model.intercepts.squeeze()

    @property
    def coefficients(self):
        return self.best_model.coefficients.squeeze()

    def predict(self, X):
        return self.best_model.predict(X).squeeze()
