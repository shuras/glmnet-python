import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from fit_and_scorers import fit_and_score_switch
from fold_generators import KFold

# Optional import: joblib.Parallel
try: 
    from sklearn.externals.joblib import Parallel, delayed 
except ImportError: 
    pass

try: 
    from joblib import Parallel, delayed
except ImportError: 
    pass

try: 
    Parallel() 
except ne: 
    par_avail = False 
else: 
    par_avail = True


def _clone(glmnet):
    '''Make a copy of an unfit glmnet object.'''
    return glmnet.__class__(**glmnet.__dict__)


class CVGlmNet(object):
    '''Manage the optimization of the lambda parameter by utilizing cross 
    validation.

      This class provides a management scheme for glmnets that abstracts 
    away the process of determining an optimimal choice of the lambda 
    parameter in a glmnet. It:

        * Provides a fit method that fits many glments over the same set of
          lambdas, but each on a random cross-fold of the training data
          provided.

    and after the CVGlmNet object has been fit:

        * Provides routines to introspect, score and visualize the resulting
          optimal model.
        * Calculate and visualize statistics pertaining to the cross validation 
          process.
    
    Both weighted and unweighted training samples are supported, and the various
    glmnet models can be fit in parallel.
    '''

    def __init__(self, glmnet, 
                 n_folds=3, n_jobs=3, include_full=True, 
                 shuffle=True, verbose=2, cv_folds=None
        ):
        '''Create a cross validation glmnet object.  Accepts the following
        arguments:

          * glmnet: 
              An object derived from the GlmNet class, currently either an
              ElasticNet or LogisticNet object.
          * cv_folds:
              A fold object. If not passed, will use n_folds and shuffle
              arguments to generate the folds.
          * n_folds: 
              The number of cross validation folds to use when fitting.
              Ignored if the cv_folds argument is given.
          * n_jobs: 
              The number of cores to distribute the work to.
          * shuffle: 
              Boolean, should the indicies of the cross validation 
              folds be shuffled randomly.
              Ignored if the cv_folds argument is given.
          * verbose: 
              Amount of talkyness.
        '''
        if cv_folds:
            self.fold_generator = cv_folds
            n_folds = cv_folds.n_folds
            shuffle = cv_folds.shuffle
        else:
            self.fold_generator = None
        if n_folds > 1 and par_avail == False:
            raise ValueError(
                "joblib.Parallel not available, must set n_folds == 1"
            )
        #if include_full:
        #    raise NotImplementedError(
        #        "include_full is not yet supported during cross validation."
        #    )
        self.base_estimator = glmnet
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.include_full = include_full
        self.shuffle = shuffle
        self.verbose = verbose

    def fit(self, X, y, weights=None, lambdas=None, **kwargs):
        '''Determine the optimal value of lambda by cross validation.

          This method fits n_fold glmnet objects, each for the same sequence 
        of lambdas, and each on a different subset of the training data. The
        resulting models are then scored on the held-out-from-fold data at each
        lambda, and the lambda that minimized the mean out of fold deviance is
        found.  This lambda is then used to fit a final model on the full
        training data.  
        '''
        self._check_if_unfit()

        if not self.fold_generator:
            self.fold_generator = KFold(
                                      X.shape[0], 
                                      n_folds=self.n_folds, 
                                      shuffle=self.shuffle,
                                      include_full=self.include_full,
                                      weights=weights,
                                  )
        # Get the appropriate fitter function
        fitter = fit_and_score_switch[self.base_estimator.__class__.__name__] 
        # Determine the sequence of lambda values to fit using cross validation.
        if lambdas is None:
            lmax = self.base_estimator._max_lambda(X, y, weights=weights)
            lmin = self.base_estimator.frac_lg_lambda * lmax  
            lambdas = np.logspace(start = np.log10(lmin),
                                  stop = np.log10(lmax),
                                  num = self.base_estimator.n_lambdas,
                      )[::-1]
        # Fit in-fold glmnets in parallel.  For each such model, pass back the 
        # series of lambdas fit and the out-of-fold deviances for each such 
        # lambda.
        models_and_devs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                              delayed(fitter)(_clone(self.base_estimator), 
                                                     X, y,
                                                     train_inds, valid_inds,
                                                     weights, lambdas,
                                                     **kwargs
                                                    )
                              for train_inds, valid_inds in self.fold_generator
                              )
        # If the full model was fit by Parallel, then pull it off 
        if self.include_full:
            self.base_estimator = models_and_devs[0][0]
        # If the full model has not been fit yet, do it now.
        if not self.base_estimator._is_fit():
            self.base_estimator.fit(X, y, weights=weights, lambdas=lambdas, **kwargs)
        # Unzip!
        models, deviances = zip(*models_and_devs[1:])
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
        # Set attributes
        self._oof_deviances = oof_deviances 
        self._oof_stds = oof_stds
        self.lambdas = lambdas
        self.best_lambda_ind = best_lambda_ind
        self.best_lambda = best_lambda

    def predict(self, X):
        '''Score the optimal model given a model matrix.'''
        self._check_if_fit()
        return self.base_estimator.predict(X)[:, self.best_lambda_ind].ravel()

    def _describe_best_est(self):
        '''Describe the optimal model.'''
        self._check_if_fit()
        return self.base_estimator.describe(lidx=self.best_lambda_ind)

    def _describe_cv(self):
        s = ("A glmnet model with optimal lambda determined by cross "
             "validation.\n"
             "The model was fit on {0} folds of the training data.\n"
             "The model was fit on {1} lambdas, with the optimal value "
             "determined to be {2}.\n").format(self.n_folds,
                                               len(self.lambdas),
                                               self.best_lambda
                                        )
        return s

    def describe(self):
        s = (self._describe_cv() + 
             '-'*79 + '\n' + "Best Model:\n" + 
             self._describe_best_est())
        return s

    def plot_oof_devs(self):
        '''Produce a plot of the mean out of fold deviance for each lambda,
        with error bars showing the standard deviation of these devinaces
        across folds.
        '''
        self._check_if_fit()
        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.lambdas)
        ax.plot(xvals, self._oof_deviances, c='blue')
        ax.scatter(xvals, self._oof_deviances, 
                   s=3, c='blue', alpha=.5
        )
        ax.errorbar(xvals, self._oof_deviances, yerr=self._oof_stds, 
                    c='grey', alpha=.5
        )
        ax.axvline(np.log(self.best_lambda), alpha=.5)
        ax.set_title("Cross validation estimates of out of sample deviance.")
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Deviance")
        plt.show()

    def _is_fit(self):
        return hasattr(self, 'best_lambda')

    def _check_if_fit(self):
        if not self._is_fit():
            raise RuntimeError("Operation is not supported with a fit model.")

    def _check_if_unfit(self):
        if self._is_fit():
            raise RuntimeError("Operation is not supported with an unfit "
                               "model."
                  )
