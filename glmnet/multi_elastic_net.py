from __future__ import absolute_import

import numpy as np
from scipy.sparse import issparse
from glmnet import _glmnet
from glmnet.glmnet import GlmNet, import_pyplot

plt = import_pyplot()

class MultiElasticNet(GlmNet):
    '''Multiresponse elastic net: a multivariate linear model with both L1 and L2
    regularizers.
    '''

    def fit(self, X, y, col_names=None,
            lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None):
        '''Fit an elastic net model.

        Arguments: 

          * X: The model matrix.  A n_obs * n_preds array.
          * y: The response.  A n_obs * n_resps array.

        Optional Arguments:
          
          * lambdas: 
              A user supplied list of lambdas, an elastic net will be fit for
              each lambda supplied.  If no array is passed, glmnet will generate
              its own array of lambdas equally spaced on a logaritmic scale 
              between \lambda_max and \lambda_min.
          * weights: 
               An n_obs array. Sample weights.
          * rel_penalties: 
              An n_preds array. Relative panalty weights for the covariates.  If
              none is passed, all covariates are penalized equally.  If an array
              is passed, then a zero indicates an unpenalized parameter, and a 1
              a fully penalized parameter.  Otherwise all covaraites recieve an
              equal penalty.
          * excl_preds: 
              An n_preds array, used to exclude covaraites from the model. To
              exclude predictors, pass an array with a 1 in the first position,
              then a 1 in the i+1st position excludes the ith covaraite from
              model fitting.  If no array is passed, all covaraites in X are 
              included in the model.
          * box_constraints:
              An array with dimension 2 * n_preds. Interval constraints on the fit
              coefficients.  The (0, i) entry is a lower bound on the ith
              covariate, and the (1, i) entry is an upper bound.  These must 
              satisfy lower_bound <= 0 <= upper_bound.  If no array is passed,
              no box constraintes are applied to the parameters.

        After fitting, the following attributes are set:
        
        Private attributes:

          * _n_fit_obs:
              The number of rows in the model matrix X.
          * _n_fit_params:
              The number of columns in the model matrix X.
          * _col_names:
              Names for the columns in the model matrix.  Used to display fit 
              coefficients.
          * _out_n_lambdas: 
              The number of lambdas associated with non-zero models (i.e.
              models with at least one none zero parameter estiamte) after
              fitting; for large enough lambda the models will become zero in
              the presense of an L1 regularizer.
          * _intecepts: 
              A one dimensional array containing the intercept estiamtes for
              each value of lambda.  See the intercepts (no underscore) 
              property for a public version.
          * _comp_coef: 
              The fit parameter estiamtes in a compressed form.  This is a
              matrix with each row giving the estimates for a single
              coefficient for various values of \lambda.  The order of the rows
              does not correspond to the order of the coefficents as given in
              the design matrix used to fit the model, this association is
              given by the _p_comp_coef attribute.  Only estaimtes that are
              non-zero for some lambda are reported.
          * _p_comp_coef: 
              A one dimensional integer array associating the coefficients in
              _comp_coef to columns in the model matrix. 
          * _indices: 
              The same information as _p_comp_coef, but zero indexed to be
              compatable with numpy arrays.
          * _n_comp_coef: 
              The number of parameter estimates that are non-zero for some
              value of lambda.
          * _n_passes: 
              The total number of passes over the data used to fit the model.
          * _error_flag: 
              Error flag from the fortran code.

        Public Attributes:

          * r_sqs: 
              An array of length _out_n_lambdas containing the r-squared
              statistic for each model.
          * out_lambdas: 
              An array containing the lambda values associated with each fit
              model.
        '''
        self._check_if_unfit()
        # Convert to arrays if native python objects
        try:
            if not issparse(X):
                X = np.asanyarray(X)
            y = np.asanyarray(y)
        except ValueError:
            raise ValueError("X and y must be either numpy arrays, or "
                             "convertable to numpy arrays."
                  )
        # Grab the design info from patsy for later use, we are abbout to write
        # over this object in some cases.
        if hasattr(X, 'design_info'):
            design_info = X.design_info
        else:
            design_info = None
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if not issparse(X) and not (np.isfortran(X) and self.overwrite_pred_ok):
            X = X.copy(order='F')
        # Make a copy if we are not able to overwrite y with its standardized
        # version.
        if not (np.isfortran(y) and self.overwrite_targ_ok):
            y = y.copy(order='F')
        # Validate all the inputs:
        self._validate_matrix(X)
        self._validate_inputs(X, y)
        self._validate_lambdas(X, y, lambdas)
        self._validate_weights(X, y, weights)
        self._validate_rel_penalties(X, y, rel_penalties)
        self._validate_excl_preds(X, y, excl_preds)
        self._validate_box_constraints(X, y, box_constraints) # dimension (2, n_
        # Setup is complete, call into the extension module.
        if not issparse(X):
            (self._out_n_lambdas,
             self._intercepts,
             self._comp_coef,
             self._p_comp_coef,
             self._n_comp_coef,
             self.r_sqs,
             self.out_lambdas,
             self._n_passes,
             self._error_flag) = _glmnet.multelnet(
                                     self.alpha, 
                                     X, 
                                     y, 
                                     self.weights, 
                                     self.excl_preds, 
                                     self.rel_penalties,
                                     self.box_constraints,
                                     self.max_vars_all, 
                                     self.frac_lg_lambda, 
                                     self.lambdas, 
                                     self.threshold, 
                                     nlam=self.n_lambdas
                                 )
        else:
            X.sort_indices()
            # Fortran arrays are 1 indexed.
            ind_ptrs = X.indptr + 1
            indices = X.indices + 1
            # Call
            (self._out_n_lambdas,
            self._intercepts,
            self._comp_coef,
            self._p_comp_coef,
            self._n_comp_coef,
            self.r_sqs,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.multspelnet(
                                    self.alpha, 
                                    X.data,
                                    ind_ptrs, 
                                    indices,
                                    y,
                                    self.weights, 
                                    self.excl_preds, 
                                    self.rel_penalties,
                                    self.box_constraints,
                                    self.max_vars_all, 
                                    self.frac_lg_lambda, 
                                    self.lambdas, 
                                    self.threshold, 
                                    nlam=self.n_lambdas
                                )
        self._check_errors()
        # The indexes into the predictor array are off by one due to fortran
        # convention differing from numpys, this make them indexes into the the
        # numpy array. 
        self._indices = np.trim_zeros(self._p_comp_coef, 'b') - 1
        # Keep some model metadata.
        self._n_fit_obs, self._n_fit_params = X.shape
        # Create a list of column names for the fit parameters, these can be
        # passed in, or attached to the matrix from patsy.  If none are found
        # we crate our own stupid ones.
        if col_names != None:
           self._col_names = col_names
        elif design_info != None:
            self._col_names = design_info.column_names
        else:
            self._col_names = [
                'var_' + str(i) for i in range(self._n_fit_params)
            ]
               
    @property
    def _coefficients(self):
        '''The fit model coefficients for each lambda.

          A _n_comp_coef * _ * _out_n_lambdas array containing the fit model
        coefficients for each value of lambda.
        '''
        self._check_if_fit()
        return self._comp_coef[:np.max(self._n_comp_coef),:,:self._out_n_lambdas]

    @GlmNet.intercepts.getter
    def intercepts(self):
        '''The fit model intercepts, one for each response for each value of lambda (n_resps x n_lambdas).'''
        self._check_if_fit()
        return self._intercepts[:, :self._out_n_lambdas]

    def deviance(self, X, y, weights=None):
        '''Calculate the normal deviance (i.e. sum of squared errors) for
        every lambda.  The model must already be fit to call this method.
        '''
        self._check_if_fit()
        if weights is not None and weights.shape[0] != X.shape[0]:
            raise ValueError("The weights vector must have the same length as X.")

        # We normalise responses by default
        resp_weights = 1.0 / np.apply_along_axis(np.nanstd, 0, np.array(y))

        y_hat = self.predict(X)
        # Take the response y, and repeat it to produce a matrix
        # of the same dimensions as y_hat
        a = np.array(y)
        y_stacked = np.tile(a.reshape(a.shape + (1,)), (1, 1, y_hat.shape[-1]))
        rw_stacked = np.tile(resp_weights.reshape(1, len(resp_weights), 1), (y_hat.shape[0], 1, y_hat.shape[2]))
        if weights is None:
            sq_residuals = ((y_stacked - y_hat) * rw_stacked)**2
            normfac = X.shape[0] * y.shape[1]
        else:
            w = np.array(weights)
            w_stacked = np.tile(w.reshape((y_hat.shape[0], 1, 1)), (1,) + y_hat.shape[1:])
            sq_residuals = w_stacked * ((y_stacked - y_hat) * rw_stacked)**2
            normfac = np.sum(weights) * y.shape[1]
        return np.apply_over_axes(np.sum, sq_residuals, [0, 1]).ravel() / normfac

    def predict(self, X):
        '''Produce model predictions from new data.
           Returns an n_obs * n_rets * n_lambdas array, where n_obs is the number of rows in X.
        '''
        self._check_if_fit()
        dot = self._get_dot(X)
        if np.max(self._n_comp_coef) > 0 :
            c = np.swapaxes(self._coefficients, 0, 1)
            return self.intercepts + dot(X[:, self._indices], c)
        else:
            return np.tile(self.intercepts, (X.shape[0], 1, 1))

    def predict_for_lambda_index(self, X, lambda_ix):
        '''Produce model predictions from new data for a given index of lambda.
           Returns an n_obs * n_rets array, where n_obs is the number of rows in X.
        '''
        self._check_if_fit()
        dot = self._get_dot(X)
        if np.max(self._n_comp_coef) > 0 :
            c = self._coefficients[:, :, lambda_ix]
            return self.intercepts[:, lambda_ix] + dot(X[:, self._indices], c)
        else:
            return np.tile(self.intercepts[:, lambda_ix], (X.shape[0], 1))

    def plot_path(self):
        '''Plot the full regularization path of all the non-zero model
        coefficients.  Creates an displays a plot of the parameter estimates
        at each value of log(\lambda).
        Betas for different response variables corresponding to every predictor are aggregated using L2 norm.
        '''
        self._check_if_fit()
        if not plt:
            raise RuntimeError('pyplot unavailable.')

        plt.clf()
        fig, ax = plt.subplots()
        xvals = np.log(self.out_lambdas[1:self._out_n_lambdas])
        def l2(x):
            return np.sqrt(np.sum(x * x))
        for coef_path in self._coefficients:
            ax.plot(xvals, np.apply_along_axis(l2, 0, coef_path[:, 1:]))
        ax.set_title("Regularization paths for multi-response elastic net with alpha = %s" %
                     (self.alpha))
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("Parameter Value")
        plt.show()


    def __str__(self):
        return self._str('multi-response elastic')

    def describe(self, lidx=None):
        return self._describe(lidx, 'mutli-response elastic')
