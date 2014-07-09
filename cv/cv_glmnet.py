import numpy as np
from sklearn.externals.joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt 
from fit_and_scorers import fit_and_score_switch
from fold_generators import unweighted_k_fold, weighted_k_fold

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
        self.base_estimator = glmnet
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.verbose = verbose

    def fit(self, X, y, weights = None, lambdas = None, **kwargs):
        '''Determine the optimal value of lambda by cross validation, and fit
        a single glmnet with this value of lambda on the full data.
        '''
        # Determine the indicies of the various train and test sets for the 
        # cross validation.
        if weights is not None:
            cv_folds = weighted_k_fold(X.shape[0], n_folds=self.n_folds,
                                                   shuffle=self.shuffle,
                                                   weights=weights
                       )
        else:
            cv_folds = unweighted_k_fold(X.shape[0], n_folds=self.n_folds, 
                                                     shuffle=self.shuffle
                       ) 
        # Copy the glmnet so we can fit the instance passed in as the final
        # model.
        base_estimator = _clone(self.base_estimator)
        fit_and_score = fit_and_score_switch[base_estimator.__class__.__name__] 
        # Determine the sequence of lambda values to fit using cross validation.
        if lambdas is None:
            lmax = base_estimator._max_lambda(X, y, weights=weights)
            lmin = base_estimator.frac_lg_lambda * lmax  
            lambdas = np.logspace(start = np.log10(lmin),
                                  stop = np.log10(lmax),
                                  num = base_estimator.n_lambdas,
                      )[::-1]
        # Fit in-fold glmnets in parallel.  For each such model, pass back the 
        # series of lambdas fit and the out-of-fold deviances for each such 
        # lambda.
        deviances = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                             delayed(fit_and_score)(_clone(base_estimator), 
                                                    X, y,
                                                    train_inds, test_inds,
                                                    weights, lambdas,
                                                    **kwargs
                                                   )
                             for train_inds, test_inds in cv_folds
                             )
        # Determine the best lambdas, i.e. the value of lambda that minimizes
        # the out of fold deviances
        dev_stack = np.vstack(deviances)
        oof_deviances = np.mean(dev_stack, axis=0)
        # TODO: Implement std dev aware strat.
        best_lambda_ind = np.argmin(oof_deviances)
        best_lambda = lambdas[best_lambda_ind]
        # Determine the standard deviation of deviances for each lambda, used
        # for plotting method.
        oof_stds = np.std(dev_stack, axis=0)
        # Refit a glmnet on the entire data set
        self.base_estimator.fit(X, y, weights=weights, lambdas=lambdas, **kwargs)
        # Set attributes
        self.oof_deviances = oof_deviances 
        self.oof_stds = oof_stds
        self.lambdas = lambdas
        self.best_lambda_ind = best_lambda_ind
        self.best_lambda = best_lambda

    def predict(self, X):
        return self.base_estimator.predict(X)[:, self.best_lambda_ind].ravel()

    def plot_oof_devs(self):
        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.lambdas)
        ax.plot(xvals, self.oof_deviances, c='blue')
        ax.scatter(xvals, self.oof_deviances, 
                   s=3, c='blue', alpha=.5
        )
        ax.errorbar(xvals, self.oof_deviances, yerr=self.oof_stds, 
                    c='grey', alpha=.5
        )
        ax.axvline(np.log(self.best_lambda), alpha=.5)
        ax.set_title("Cross validation estimates of out of sample deviance.")
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Deviance")
        plt.show()
