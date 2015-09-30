import numpy as np

class KFold(object):
    '''Manages cross validation fold indices.

      This calss exposes an interface for creating iteratiors that yield
    indices of cross validation folds.  The canonical usage goes like this:

        kf = KFold(n=<number of samples, n_folds=<desired number of folds>)
        for train_idxs, valid_idxs in kf:
             train_data_for_fold = data[train_idxs,:]
             validation_data_for_fold = data[valid_idxs,:]

    The class conforms to the api of sklearn.cross_validation.KFold, but also
    allows the user to supply sample weights, in this case the generated 
    folds will be balanced by total weight instead of number of samples.

      The edge cases are balanced so that in the case where n_folds divides n
    exactly, each fold will contain exactly n / n_folds samples.
    '''

    def __init__(self, n, n_folds,
                 weights=None, shuffle=False, include_full=False):
        '''Create a CVfold object.  The following arguments are accepted:

            * n: 
                The number of samples in the data set being folded.
            * n_folds:
                The number of folds to create.
            * weights:
                Sample weights for the data being folded.  If provides the
                generated folds are balanced by total weight.
            * shuffle:
                Randomize the sample indecies before creating folds.  If not
                supplied the validation fold indicies generated will always be 
                consecutive. 
            * include_full:
                Include the full data set as a generated training fold, with
                the corresponding validation fold empty.
        '''
        self.n = n
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.weights = weights if weights != None else np.ones(shape=n)
        self.include_full = include_full

    def __iter__(self):
        '''Generator for producing indices into the training data correspoding
        to folds for cross validation.  Yields pairs of arrays:

            (<array of indicies into the data representing the training data
              for this fold.>,
             <array of indicies into the data representing the validation data
              for this fold.>
            )

          The algorithm for producing folds of equal weight is the following.

              * Shuffle the weights vector if requested.
              * Compute an array representing the cumulative density function
                of the weights.
              * Compute the n_folds order statistics of the resulting cdf. 
                That is, find the indicies into the cdf array that most closely
                partitions it into n_folds of equal weight.
              * For each order statistic os(k), yield the indicies corresponding
                to the weights between the statistics os(k-1) and os(k) as the
                validation indicies, and the complementary indicies as the train.

        Additionally, if the full data set was requested as a fold, we yield 
        all indicies as a training fold and an empty validation fold as our 
        first pair yielded to the caller.
        '''
        weights = np.asarray(self.weights)
        n = self.n
        if weights.shape[0] != n:
            raise ValueError("Weights must have length n.")
        samples = np.arange(n)
        # Include the full data as a training fold if requested to do so
        if self.include_full:
            yield samples, np.array([]).astype(np.int64)
        # Randomize the order of the training indices if requested to do so
        if self.shuffle:
            np.random.shuffle(samples)
            pdf = weights[samples]
        else:
            pdf = weights
        # Compute and yield the folds
        cdf = np.cumsum(pdf)
        wsum = np.sum(weights)
        cutoffs = np.linspace(0, wsum, self.n_folds + 1)
        for i in range(1, self.n_folds + 1):
            valid_interval = (cdf > cutoffs[i-1])*(cdf <= cutoffs[i])
            train_interval = 1 - valid_interval
            valid_inds = samples[np.nonzero(valid_interval)]
            train_inds = samples[np.nonzero(train_interval)]
            yield train_inds, valid_inds

    def _make_equal_weights(self):
        return np.ones(shape=self.n)
