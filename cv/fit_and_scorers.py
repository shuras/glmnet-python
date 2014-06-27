def fit_and_score_elastic_net(elastic_net, X, y, 
                               train_inds, test_inds, 
                               **kwargs):
    if 'weights' in kwargs:
        weights = kwargs['weights']
        train_weights = weights[train_inds]
        test_weights = weights[test_inds]
        del kwargs['weights']
    else:
        train_weights, test_weights = None, None
    elastic_net.fit(X[train_inds], y[train_inds], 
                    weights=train_weights, **kwargs
                )
    return (elastic_net.out_lambdas[:elastic_net._out_n_lambdas], 
            elastic_net.deviance(X[test_inds], y[test_inds], 
                                 weights=test_weights
                        )
           )

def fit_and_score_logistic_net(logistic_net, X, y, 
                                train_inds, test_inds,
                                **kwargs):
    logistic_net.fit(X[train_inds], y[train_inds], **kwargs)
    return (logistic_net.out_lambdas[:logistic_net._out_n_lambdas], 
            logistic_net.deviance(X[test_inds], y[test_inds])
           )

fit_and_score_switch = {'ElasticNet': fit_and_score_elastic_net,
                        'LogisticNet': fit_and_score_logistic_net
                       }
