def fit_and_score_elastic_net(elastic_net, X, y, 
                              train_inds, test_inds, 
                              weights, lambdas,
                              **kwargs):
    if weights is not None:
        train_weights = weights[train_inds]
        test_weights = weights[test_inds]
    else:
        train_weights, test_weights = None, None
    elastic_net.fit(X[train_inds], y[train_inds], 
                    weights=train_weights, lambdas=lambdas,
                    **kwargs
                )
    return elastic_net.deviance(X[test_inds], y[test_inds], 
                                 weights=test_weights
                       )

def fit_and_score_logistic_net(logistic_net, X, y, 
                               train_inds, test_inds,
                               weights, lambdas, 
                               **kwargs):
    logistic_net.fit(X[train_inds], y[train_inds], lambdas=lambdas, **kwargs)
    return logistic_net.deviance(X[test_inds], y[test_inds])

fit_and_score_switch = {'ElasticNet': fit_and_score_elastic_net,
                        'LogisticNet': fit_and_score_logistic_net
                       }
