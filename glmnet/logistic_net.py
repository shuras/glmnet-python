import numpy as np
from scipy.sparse import issparse
import _glmnet
from glmnet import GlmNet

class LogisticNet(GlmNet):
    '''The logistic net: a multivariate logistic model with both L1 and L2
    regularizers.

      This class implements the logistic net class of predictive models. These
    models combine the classical ridge and lasso regularizers into a combined
    penalty to the binomial deviance loss function.  More specifically, the
    loss function minimized by this model is:

        L(\beta_0, \beta_1, ..., \beta_n) =
            BinDev(\beta_0, \beta_1, ..., \beta_n; X, y) + 
            \lambda * ((\alpha - 1)/2 * | \beta |_2 + \alpha * | \beta |_1)

    where BinDev is the usual binomial deviance:

      BinDev(\beta_0, \beta_1, ..., \beta_n; X, y) =
       -2*sum( y*log(1 - p) + (1 - y)*log(1 - p) )

    in which p is formed by applying the usual logistic function to a linear
    predictor.
    '''

    def fit(self, X, y, col_names=None,
            lambdas=None, weights=None, rel_penalties=None,
            excl_preds=None, box_constraints=None, offsets=None):
        '''Fit a logistic or multinomial net model.

        Arguments:

          * X: The model matrix.  A n_obs * n_preds array.
          * y: The response.  This method accepts the response in two
            differnt configurations:

            - An n_obs * n_classes array.  In this case, each column in y must
              be of boolean (0, 1) type indicating whether the observation is
              or is not of a given class.
            - An n_obs array.  In this case the array must contain a discrete
              number of values, and is converted into the previous form before
              being passed to the model.

        Optional Arguments:

          * lambdas: 
              A user supplied list of lambdas, an elastic net will be fit for
              each lambda supplied.  If no array is passed, glmnet will generate
              its own array of lambdas equally spaced on a logaritmic scale 
              between \lambda_max and \lambda_min.
          * weights: 
               An n_obs array. Sample weights. It is an error to pass a weights
               array to a logistic model.
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
              An array with dimension 2 * n_obs. Interval constraints on the fit
              coefficients.  The (0, i) entry is a lower bound on the ith
              covariate, and the (1, i) entry is an upper bound.  These must 
              satisfy lower_bound <= 0 <= upper_bound.  If no array is passed,
              no box constraintes are allied to the parameters.
          * offsets: 
              A n_preds * n_classes array. Used as initial offsets for the
              model fitting. 

        After fitting, the following attributes are set:
        
        Private attributes:

          * _n_fit_obs:
              The number of rows in the model matrix X.
          * _n_fit_params:
              The number of columns in the model matrix X.
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
          * _indicies: 
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
          
          * null_dev: 
              The devaince of the null (mean) model.
          * exp_dev: 
              The devaince explained by the model.
          * out_lambdas: 
              An array containing the lambda values associated with each fit
              model.
        '''
        self._check_if_unfit()
        self._check_y(y)
        self._check_weights(weights)
        # Convert to arrays is native python objects
        try:
            if not issparse(X):
                X = np.asanyarray(X)
            y = np.asanyarray(y)
        except ValueError:
            raise ValueError("X and y must be wither numpy arrays, or "
                             "convertable to numpy arrays."
                  )
        # Fortran expects an n_obs * n_classes array for y.  If a one 
        # dimensional array is passed, we construct an appropriate widening. 
        # TODO: Save class names as attribute.
        y = np.asanyarray(y)
        if len(y.shape) == 1:
            y_classes = np.unique(y)
            y = np.float64(np.column_stack(y == c for c in y_classes))
        self._n_classes = y.shape[1]
        # Two class predictions are handled as a special case, as is usual 
        # with logistic models, this is signaled to fortran by passing in a
        # 1 for nc (num classes).
        if self._n_classes == 2:
            f_n_classes = np.array([1])
        else:
            f_n_classes = np.array([self._n_classes])
        # Grab the design info from patsy for later use, we are abbout to write
        # over this object in some cases.
        if hasattr(X, 'design_info'):
            design_info = X.design_info
        else:
            design_info = None
        # Make a copy if we are not able to overwrite X with its standardized 
        # version. Note that if X is not fortran contiguous, then it will be 
        # copied anyway.
        if not issparse(X) and np.isfortran(X) and not self.overwrite_pred_ok:
            X = X.copy(order='F')
        # Make a copy if we are not able to overwrite y with its standardized
        # version.
        if np.isfortran(y) and not self.overwrite_targ_ok:
            y = y.copy(order='F')
        # Validate all the inputs:
        self._validate_matrix(X)
        self._validate_inputs(X, y)
        self._validate_lambdas(X, y, lambdas)
        self._validate_rel_penalties(X, y, rel_penalties)
        self._validate_excl_preds(X, y, excl_preds)
        self._validate_box_constraints(X, y, box_constraints)
        self._validate_offsets(X, y, offsets)
        # Setup is complete, call the wrapper
        if not issparse(X):
            (self._out_n_lambdas,
            self._intercepts,
            self._comp_coef,
            self._p_comp_coef,
            self._n_comp_coef,
            self.null_dev,
            self.exp_dev,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.lognet(
                                    self.alpha, 
                                    f_n_classes,
                                    X,
                                    y, 
                                    self.offsets,
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
            self.null_dev,
            self.exp_dev,
            self.out_lambdas,
            self._n_passes,
            self._error_flag) = _glmnet.splognet(
                                    self.alpha, 
                                    X.shape[0],
                                    X.shape[1],
                                    f_n_classes,
                                    X.data,
                                    ind_ptrs,
                                    indices,
                                    y, 
                                    self.offsets,
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
        # Keep some model metadata
        self._n_fit_obs, self._n_fit_params = X.shape
        # The indexes into the predictor array are off by one due to fortran
        # convention differing from numpys, this make them indexes into the the
        # numpy array. 
        self._indicies = np.trim_zeros(self._p_comp_coef, 'b') - 1
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

    def _validate_offsets(self, X, y, offsets):
        '''If no explicit offset was setup, we assume a zero offset.'''
        # TODO: Allow for passing a one-dim array of offsets in the two 
        # class case.
        if offsets is None:
            self.offsets = np.zeros((X.shape[0], y.shape[1]), order='F')
        else:
            self.offsets = offsets
        if (self.offsets.shape[0] != X.shape[0] or
            self.offsets.shape[1] != y.shape[1]):
            raise ValueError("Offsets must share its first dimenesion with X "
                             "and its second dimension must be the number of "
                             "response classes."
                  )

    @property
    def _coefficients(self):
        '''The fit model coefficients for each lambda.
 
          The dimensions of this array depend on whether of not a two class
        response was used (i.e. was logistic regression performed):

          * If so: 
              A _n_comp_coef * _out_n_lambdas array containing the fit model
              coefficients for each value of lambda.

          * If not: 
              A _n_comp_coef * n_classes * _out_n_lambdas array containing the
              model coefficients for each level of the response, for each
              lambda.
        '''
        self._check_if_fit()
        return self._comp_coef[:np.max(self._n_comp_coef),
                               :,
                               :self._out_n_lambdas
                    ].squeeze()

    def _max_lambda(self, X, y, weights=None):
        '''Return the maximum value of lambda useful for fitting, i.e. that
        which first forces all coefficients to zero.

          This calculation is derived from the discussion in "Regularization 
        Paths for Generalized Linear Models via Coordinate Descent" in the 
        section "Regularized Logistic Regression", using an analogy with the 
        linear case with weighted samples.  
        
          We apply the reasoning from the linear case to a quadratic 
        approximation to the binomial likelihood evaluated at an initial 
        value of the parameters.  The initial parameters are taken to be:

            \beta_0 = log(p/(1-p))
            \beta_i = 0

        I.e., our initial model is the intercept only model.  Comapring the
        approximation to the elastic net case gives us a working response and
        working weights.  These are used in the formula giving the maximum
        lambda in the elastic net.
        
           There is one complication: the working weights in the quadratic 
        approximation to the logistic loss are not normalized, while the 
        discussion in the linear case makes this assumption.  To compensate
        for this, we must adjust the value of lambda to live on the same
        scale as the weights, which causes the normalization factor to drop
        out of the equation.
        '''
        # TODO: These calculations assume the y is a one dimensional vector 
        # of 0s and 1s, they do not cover the case of a multi column y.
        self._check_y(y)
        if issparse(X):
            return self._max_lambda_sparse(X, y, weights)
        else:
            return self._max_lambda_dense(X, y, weights)

    def _max_lambda_dense(self, X, y, weights=None):
        self._check_y(y)
        self._check_weights(weights)
        # Standardize the model matrix.
        normfac = X.shape[0]
        mu = X.sum(axis=0) / normfac
        mu2 = (X*X).sum(axis=0) / normfac
        X_scaled = (X - mu) / np.sqrt(mu2 - mu*mu)
        dots = np.dot(y, X_scaled)
        # Working response
        p = np.mean(y)
        working_resp = np.log(p/(1-p)) + (y - p) / (p*(1 - p))
        # Working weights
        working_weights = p*(1 - p) / (X.shape[0])
        # Now mimic the calculation for the quadratic case
        y_wtd = working_resp * working_weights
        dots = y_wtd.dot(X_scaled)
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / alpha

    def _max_lambda_sparse(self, X, y, weights=None):
        '''To preserve the sparsity, we must avoid explicitly subtracting out
        the mean of the columns.
        '''
        if weights is not None:
            raise ValueError("LogisticNet cannot be fit with weights.")
        # Sparse dot product
        dot = self._get_dot(X)
        # Calculation is modeled on weighted least squares
        p = np.mean(y)
        working_resp = np.log(p/(1-p)) + (y - p) / (p*(1 - p))
        working_weights = p*(1 - p) / (X.shape[0])
        # Sample expectataion value of the columns in a matrix
        E = lambda M: np.asarray(M.sum(axis=0)).ravel() / M.shape[0]
        mu = E(X)
        mu_2 = E(X.multiply(X))
        sigma = np.sqrt(mu_2 - mu*mu)
        # Calculating the dot product of y with X standardized, without 
        # destorying the sparsity of X
        y_wtd = working_resp * working_weights
        dots = 1/sigma * (dot(y_wtd, X) - mu * np.sum(y_wtd))
        # An alpha of zero (ridge) breaks the maximum lambda logic, the 
        # coefficients are never all zero - so we readjust to a small
        # value.
        alpha = self.alpha if self.alpha > .0001 else .0001
        return np.max(np.abs(dots)) / alpha

    def deviance(self, X, y):
        '''Calculate the binomial deviance for every lambda. The model must
        already be fit to call this method.
        '''
        # TODO: This only covers the case of a one dimensional y, need to 
        # support multiple columns for y.
        y_hat = self.predict(X)
        # Take the response y, and repeat it as a column to produce a matrix
        # of the same dimensions as y_hat
        y_stacked = np.tile(np.array([y]).transpose(), y_hat.shape[1])
        #print y_stacked
        bin_dev = y_stacked*np.log(y_hat) + (1 - y_stacked)*np.log(1 - y_hat)
        normfac = X.shape[0]
        return np.apply_along_axis(np.sum, 0, -2*bin_dev) / normfac

    def predict(self, X):
        '''Return model predictions on the probability scale.'''
        # TODO This only support a one dimensional y.
        return 1 / ( 1 + np.exp(self._predict_lp(X)) )

    def plot_paths(self):
        self._plot_path('logistic')

    def _check_y(self, y):
        '''Temporary safety net.  glmnet has the ability to fit a mutinomial
        model, but this class currently cannot handle that case.
        '''
        if len(y.shape) != 1:
            raise NotImplementedError(
                "This method does not yet support non-one dimensional y "
                "arrays."
            )

    def _check_weights(self, weights):
        '''Logistic models do not support sample weights.'''
        if weights is not None:
            raise ValueError("LogisticNet cannot be fit with weights.")

    def __str__(self):
        return self._str('logistic')

    def describe(self, lidx=None):
        return self._describe('logistic')
