'''A module containing utility routines to:

    * fit a glmnet model on a subset of a training data set.
    * score a glmnet model on a subset of a valation set.
    * return the deviances as estiamted using the validation set.

these are indended to be mapped across a specification of cross validation
folds in cv_glmnet.
'''

def fit_and_score_elastic_net(elastic_net, X, y, 
                              train_inds, valid_inds, 
                              weights, lambdas,
                              **kwargs):
    if weights is not None:
        train_weights = weights[train_inds]
        valid_weights = weights[valid_inds]
    else:
        train_weights, valid_weights = None, None
    elastic_net.fit(X[train_inds], y[train_inds], 
                    weights=train_weights, lambdas=lambdas,
                    **kwargs
                )
    if valid_inds.shape[0] != 0:
        valid_dev = elastic_net.deviance(X[valid_inds], y[valid_inds], 
                                         weights=valid_weights
                        )
    else:
        valid_dev = None

    return (elastic_net, valid_dev)

def fit_and_score_logistic_net(logistic_net, X, y, 
                               train_inds, valid_inds,
                               weights, lambdas, 
                               **kwargs):
    logistic_net.fit(X[train_inds], y[train_inds], lambdas=lambdas, **kwargs)
    if valid_inds.shape[0] != 0:
        valid_dev = logistic_net.deviance(X[valid_inds], y[valid_inds])
    else:
        valid_dev = None

    return (logistic_net, valid_dev)

fit_and_score_switch = {'ElasticNet': fit_and_score_elastic_net,
                        'LogisticNet': fit_and_score_logistic_net,
                        'MultiElasticNet': fit_and_score_elastic_net
                       }
