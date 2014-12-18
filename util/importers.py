'''Handles optional dependencies.'''
from warnings import warn

def import_joblib(verbose=1):

    try: 
        import sklearn.externals.joblib as jl
    except ImportError: 
        jl = None

    if not jl:
        try:
            import joblib as jl
        except ImportError:
            jl = None

    if not jl and verbose > 0:
        warn('joblib not found, parallel fitting of models will be '
             'unavailable.', ImportWarning)

    return jl


def import_pyplot(verbose=1):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if not plt and verbose > 0:
        warn('pyplot unavailabe, plotting functionality will be unavaliabe.',
             ImportWarning)

    return plt
