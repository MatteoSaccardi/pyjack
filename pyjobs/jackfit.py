import numpy
import matplotlib.pyplot as plt
# import scipy
# import pyobs
from .pyjack import observable
import sympy
from typing import Union

from .utils import plt_errorbar_fill_color

plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def svd_sqrt(A):
    U,S,V = numpy.linalg.svd(A)
    return (V.conj().T * numpy.sqrt(S)) @ U.conj().T

def svd_inv(A):
    U,S,V = numpy.linalg.svd(A)
    return (V.conj().T * (1.0/S)) @ U.conj().T

class LevenbergMarquardt:
    '''
    Levenberg-Marquardt algorithm for non-linear least-squares minimization
    '''
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray, model: callable, jacobian: callable, W2: numpy.ndarray, initial_guess: numpy.ndarray, max_iter: int = 1000, tol: float = 1e-8):
        '''
        Initialize and perform Levenberg-Marquardt minimization

        Parameters:
        - x: Independent variable values.
        - y: Observed dependent variable values.
        - model: Function representing the model to fit.
        - jacobian: Function computing the Jacobian matrix.
        - W2: Weighting matrix (e.g. inverse of covariance matrix or its diagonal).
        - initial_guess: Initial parameter estimates.
        - max_iter: Maximum number of iterations (default: 1000).
        - tol: Convergence tolerance (default: 1e-8).
        '''
        self.success: bool = 0
        self.message: str = ''
        self.x: Union[numpy.ndarray, None] = None
        self.initial_guess: numpy.ndarray = numpy.array(initial_guess, dtype=float)
        self.minimize(x, y, model, jacobian, W2, initial_guess, max_iter, tol)
    def minimize(self, x: numpy.ndarray, y: numpy.ndarray, model: callable, jacobian: callable, W2: numpy.ndarray, initial_guess: numpy.ndarray, max_iter: int = 1000, tol: float = 1e-8):
        message = f'[LevenbergMarquardt.minimize] Maximum number of iterations reached, namely {max_iter}, with tolerance {tol}. Exiting without convergence'
        success = 0
        params = numpy.array(initial_guess, dtype=float)
        lambda_ = 1e-3 # Initial damping factor
        for i in range(max_iter):
            residuals = y - model(x, *params)
            J = numpy.array(jacobian(x, *params)).T
            H = J.T @ W2 @ J
            A = H + lambda_ * numpy.eye(len(params))#numpy.diag(numpy.diag(H))
            Ainv = svd_inv(A)
            b = J.T @ W2 @ residuals
            delta_p = Ainv @ b #numpy.linalg.solve(A, b)
            new_params = params + delta_p
            # Check if improvement is significant
            new_residuals = y - model(x, *new_params)
            if numpy.linalg.norm(new_residuals) < numpy.linalg.norm(residuals):
                params = new_params
                lambda_ /= 10 # Reduce damping
            else:
                lambda_ *= 10 # Increase damping
            if numpy.linalg.norm(delta_p) < tol:
                message = f'[LevenbergMarquardt.minimize] Convergence with tolerance {tol} reached after {i} iterations. Exiting successfully'
                success = 1
                break
        self.success = success
        self.message = message
        self.x = params

def chi2(params, x, y, model, W2):
    '''
    Compute chi-squared given model function and matrix W as
        \sum_{x1,x2} [y(x1)-model(x1)] W2(x1,x2) [y(x2)-model(x2)]
    '''
    res = y - model(x, *params)
    return res.T @ W2 @ res

def chi2_residuals(params, x, y, model, W2):
    '''
    Compute chi-squared residuals for least-squares fitting:
        r(x) = sqrt(W2) * (y - model(x))
    '''
    res = y - model(x, *params)
    return svd_sqrt(W2) @ res
    
def expr_to_func(expr):
    '''
    Convert a string expression into a model function and its Jacobian matrix.
    
    Parameters:
    - expr : str
        A string representing the model function. The model should be expressed 
        in terms of parameters labeled as `params0`, `params1`, etc.
        Example: `'params0 * exp(-params1 * x)'`.
    
    Returns:
    - f_sympy : callable
        A callable model function using sympy.
    - f_numpy : callable
        A callable model function using numpy.
    - J : callable
        A callable function computing the Jacobian matrix.
    '''
    x = sympy.symbols('x')
    # Extract unique parameter symbols
    expr_sympy = sympy.sympify(expr)
    params = sorted([s for s in expr_sympy.free_symbols if 'params' in s.name], key=lambda s: s.name)
    f_sympy = sympy.lambdify((x, *params), expr_sympy, 'sympy')
    f_numpy = sympy.lambdify((x, *params), expr_sympy, 'numpy')
    # Compute Jacobian
    jacobian = [sympy.diff(expr_sympy, param) for param in params]
    J = sympy.lambdify((x, *params), jacobian, 'numpy')
    return f_sympy, f_numpy, J

def expr_to_pyobs(expr):
    '''
    Converts an expression by making it suitable for a pyobs function call:
    1. Replacing standalone elementary functions (exp, log, sin, etc.) with `pyobs.*`
       while keeping `numpy.*` or `math.*` prefixed functions unchanged.
    2. Replacing `params0`, `params1`, ... with `params[0]`, `params[1]`, ...
       after modifying function names.
    
    Parameters:
        expr (str): A mathematical expression as a string.

    Returns:
        str: The modified expression.
    '''
    function_map = {
        'exp': 'pyobs.exp',
        'log': 'pyobs.log',
        'sin': 'pyobs.sin',
        'cos': 'pyobs.cos',
        'tan': 'pyobs.tan',
        'sqrt': 'pyobs.sqrt',
        'arcsin': 'pyobs.arcsin',
        'arccos': 'pyobs.arccos',
        'arctan': 'pyobs.arctan',
        'cosh': 'pyobs.cosh',
        'sinh': 'pyobs.sinh',
        'tanh': 'pyobs.tanh',
        'arcsinh': 'pyobs.arcsinh',
        'arccosh': 'pyobs.arccosh'
    }
    result = []
    i = 0
    while i < len(expr):
        # Check if the current substring is a function name
        for func in function_map:
            if expr[i:].startswith(func): # Found a function
                # Check if it's prefixed by 'numpy.' or '.pyobs'
                prefix_start = max(0, i - 6) # Look back up to 6 characters
                prefix = expr[prefix_start:i]
                if not (prefix.endswith('numpy.') or prefix.endswith('pyobs.')):
                    result.append(function_map[func]) # Replace function
                    i += len(func) # Move index forward
                    break
        else:
            result.append(expr[i])
            i += 1 # Move forward by one character
    modified_expr = ''.join(result)
    # Step 2: Replace paramsX (e.g., params0, params1) with params[X]
    result = []
    i = 0
    while i < len(modified_expr):
        if modified_expr[i:].startswith('params'):
            j = i + 6 # Move past 'params'
            while j < len(modified_expr) and modified_expr[j].isdigit():
                j += 1 # Collect digits
            param_index = modified_expr[i+6:j] # Extract number
            result.append(f'params[{param_index}]') # Replace with params[index]
            i = j  # Move index forward
        else:
            result.append(modified_expr[i])
            i += 1 # Move forward by one character
    modified_expr = ''.join(result)
    # Return a lambda function of (x, params)
    return eval(f'lambda x, params: {modified_expr}', {'pyobs': pyobs, 'numpy': numpy})

class jackfit:
    def __init__(self, expr_model: str, W2: Union[numpy.ndarray, str], initial_guess):
        '''
        Initialize the jackfit class
        
        Parameters:
        - expr_model : str
            A string representing the model function. The model should be expressed 
            in terms of parameters labeled as `params0`, `params1`, etc.
            Example: `'params0 * exp(-params1 * x)'`.
        - W2 : str or array-like
            The weighting matrix for the chi-squared minimization. 
            -> `'diag'`: Uses the diagonal elements of the covariance matrix (uncorrelated errors).
            -> `'full'`: Uses the full covariance matrix (correlated errors).
            -> If an explicit array is provided, it is used as is; if 1d, taken as the diagonal of a 2d matrix.
        - initial_guess : list or array-like
            Initial guess for the fit parameters. The length must match the number of parameters in `expr`.
        '''
        self.expr_model = expr_model
        self.W2 = W2
        self.initial_guess = numpy.array(initial_guess, dtype=float)
        independent_params = sorted([s for s in sympy.sympify(expr_model).free_symbols if 'params' in s.name], key=lambda s: s.name)
        if len(initial_guess) != len(independent_params):
            raise ValueError('[jackfit.fit] The model you use must be defined in terms of an expression containing parameters in the form of params#, e.g. params0 * exp(-params1*x), in the same number as the length of the initial_guess for fit parameters!')
        sym_model, self.model, jacobian = expr_to_func(expr_model)
        def array_jacobian(x,*params):
            try:
                x = float(x)
            except:
                x = numpy.array(x)
            jac_list = jacobian(x, *params)
            for ij,j in enumerate(jac_list):
                if isinstance(j,(float,int)) and not isinstance(x,(float,int)):
                    jac_list[ij] = numpy.array( [j] * x.shape[0] )
            return jac_list
        self.jacobian = array_jacobian
        # self.pyobs_model = expr_to_pyobs(expr_model)
        self.params = None
        self.P = None
        self.chi2exp = None
        self.chi2obs = [None,None]
        self.pvalue = [None,None]
    
    def __call__(self, x: numpy.ndarray, plot=False, cov=False, log=False):
        return self.extrapolate(x,plot,cov,log)
    
    def chi2(self, params, x, y):
        '''
        Compute chi-squared given model function and matrix W as
            \sum_{x1,x2} [y(x1)-model(x1)] W2(x1,x2) [y(x2)-model(x2)]
        '''
        res = y - self.model(x, *params)
        return res.T @ self.W2 @ res
    
    def chi2_residuals(self, params, x, y):
        '''
        Compute chi-squared residuals for least-squares fitting:
            r(x) = W (y - model(x)), W = sqrt(W2)
        '''
        res = y - self.model(x, *params)
        return self.W @ res
    
    def compute_proj(self, W2=None, J=None):
        if W2 is None:
            W2 = self.W2
            W = self.W
        else:
            W = svd_sqrt(W2)
        if J is None:
            J = self.J
        # to compute W (1-P) W, we do not need W = sqrt(W2) -> avoid it
        # W2 has shape (T,T), J has shape (Npars,T)
        H = J @ W2 @ J.T # shape (Npars,Npars) # Eq. (A.2)
        self.constH = H[0,0] # work with O(1) numbers
        self.Hinv = svd_inv(H/self.constH)
        # EVERY TIME THERE IS Hinv, MULTIPLY BY 1/self.constH BACK
        # e.g. by absorbing it in W or W2
        W2J = W2 @ J.T
        self.W1PW = W2 - (W2J/numpy.sqrt(self.constH)) @ self.Hinv @ (W2J.T/numpy.sqrt(self.constH))
        # compute P (just to check, as it will never be explicitly needed)
        WJ = W @ J.T / numpy.sqrt(self.constH) # shape (T,Npars)
        self.P = WJ @ self.Hinv @ WJ.T # Eq. (A.4)
        # check projector properties in Eq. (2.9)
        # i.e. P WJ = WJ, P^2 = P, tr(P) = Npars
        check = numpy.linalg.norm(self.P@WJ-WJ) / numpy.linalg.norm(WJ)
        if check > 1e-12:
            print(f'[jackfit.compute_proj] PWJ=WJ does not hold: ||PWJ-WJ|| / ||WJ||={check}')
        check = numpy.linalg.norm(self.P@self.P-self.P) / numpy.linalg.norm(self.P)
        if check > 1e-12:
            print(f'[jackfit.compute_proj] P^2=P does not hold: ||PP-P|| / ||P||={check}')
        if numpy.fabs(numpy.sum(numpy.diag(self.P))-WJ.shape[1]) > 1e-10:
            print(f'[jackfit.compute_proj] tr[P] = {numpy.sum(numpy.diag(self.P))} != Npars = {WJ.shape[1]}')
        
    def compute_chi2exp(self, cov=None, W2=None, J=None):
        # Eq. (2.13)
        if cov is None:
            cov = self.cov
        if W2 is None:
            if (J is not None) or (J is None and self.W1PW is not None):
                self.compute_proj(W2,J)
        else:
            self.compute_proj(W2,J)
        W1PW = self.W1PW # W @ (numpy.eye(W.shape[0]) - self.P) @ W
        chi2exp = numpy.sum(numpy.diag(cov @ W1PW))
        try:
            # ncnfg = self.obs.delta[list(self.obs.delta.keys())[0]].ncnfg() # pyobs
            ncnfg = self.obs.N
        except:
            ncnfg = 1
        # err_chi2exp = numpy.sqrt( 2 * (2*self.errinfo[self.tag].W+1) / ncnfg ) * chi2exp # Eq. (3.11) # pyobs
        err_chi2exp = numpy.sqrt( 2 * (2*0+1) / ncnfg ) * chi2exp
        return chi2exp, err_chi2exp
    
    def compute_chi2expMB(self, W2=None, J=None):
        if W2 is None:
            if (J is not None) or (J is None and self.W1PW is not None):
                self.compute_proj(W2,J)
        else:
            self.compute_proj(W2,J)
        w, v = numpy.linalg.eig(self.W1PW) #W @ (numpy.eye(W.shape[0]) - self.P) @ W
        c2exp = self.obs @ v
        # _, ce, dce = c2exp.error_core(plot=False, errinfo=self.errinfo, pfile=None) # pyobs
        ce, dce = c2exp.mean(), c2exp.error()
        return w @ ce, w @ dce
        
    def compute_cov_params(self, cov=None, W2=None, J=None):
        # cov and W have shape (T,T), J has shape (Npars,T)
        if cov is None:
            cov = self.cov
        if W2 is None:
            if (J is not None) or (J is None and self.W1PW is not None):
                self.compute_proj(W2,J)
        else:
            self.compute_proj(W2,J)
        if J is None:
            J = self.J
        if W2 is None:
            W2 = self.W2
        # Also here, for cov_params in Eq. (A.3), W = sqrt(W2) is never needed
        #WJ = W @ J.T # shape (T,Npars)
        #Wr = W / numpy.sqrt(self.constH)
        #WJ /= numpy.sqrt(self.constH)
        #cov_params = self.Hinv @ (WJ.T @ Wr @ cov @ Wr @ WJ) @ self.Hinv # Eq. (A.3)
        cov_params = self.Hinv @ J @ (W2/self.constH) @ cov @ (W2/self.constH) @ J.T @ self.Hinv
        return cov_params

    def compute_pvalue(self, chi2obs=None, cov=None, W2=None, J=None, num_samples=10000):
        '''
        Compute, using Monte-Carlo integration, the p-value as
               Q(\chi^2_{\obs},\nu) 
               = \int \prod_{i=1}^{N_\nu} dz_i P(z_i)
               \theta( \sum_{j=1}^{N_\nu} \lambda_j(\nu) z_j^2 - \chi^2_{obs} )
        where P(z_i) = exp{-z_i^2/2} / \sqrt{2\pi}
        '''
        if chi2obs is None:
            chi2obs = self.chi2obs
        if cov is None:
            cov = self.cov
        if W2 is None:
            if (J is not None) or (J is None and self.W1PW is not None):
                self.compute_proj(W2,J)
        else:
            self.compute_proj(W2,J)
        # Compute nu from Eq. (2.17)
        #W1PW = W @ (numpy.eye(W.shape[0])-self.P) @ W
        #sqrt_cov = svd_sqrt(self.cov)
        #nu = sqrt_cov @ W1PW @ sqrt_cov
        # As we only want to compute the eigenvalues of nu, it suffices to compute those of
        lambdas = numpy.linalg.eigvalsh(cov @ self.W1PW)
        lambdas = lambdas[lambdas > 0]
        N_nu = len(lambdas)
        if N_nu == 0:
            return 0, 0
        z = numpy.random.randn(num_samples,N_nu)
        # Rather than
        sum_lambda_z2 = (z**2) @ lambdas # numpy.sum(lambdas*z**2,axis=1)
        theta = sum_lambda_z2 >= chi2obs # numpy.where(sum_lambda_z2 >= chi2obs, 1, 0)
        # using <z_i^2>=1 i.e. <sum_i z_i^2>=N_nu, we rather employ
        #sum_lambda_z2_chi2obs = (z**2) @ (lambdas-chi2obs/N_nu)
        #theta = sum_lambda_z2_chi2obs >= 0 # numpy.where(sum_lambda_z2_chi2obs >= 0, 1, 0)
        p = numpy.mean(theta)
        dp = numpy.std(theta,ddof=1) / numpy.sqrt(num_samples)
        return p, dp
        
    def fit(self, x: numpy.ndarray, obs: observable, max_iter: int = 1000, tol: float = 1e-8, num_samples: int = 10000, cov = None, mean = None):
        '''
        Perform a chi-squared minimization fit using the Levenberg-Marquardt algorithm.

        Parameters:
        -----------
        x : array-like
            The independent variable values.
        obs : pyjack observable
            A `pyjack` observable containing the measured data. 
            The function extracts:
            - `obs.error()` for uncertainties on the measurements.
            - `obs.covariance_matrix()` for the covariance matrix.
        max_iter : int = 1000
            Maximum number of iterations for Levenberg-Marquardt minimizer.
        tol : float = 1e-8
            Tolerance for Levenberg-Marquardt minimizer.
        num_samples : int = 10000
            Number of samples to estimate the p-value.
        cov : array-like
            If provided, takes precedence over `obs.covariance_matrix(errinfo=errinfo)[0]` to define the covariance matrix.
        mean : array-like
            If provided, takes precedence over `obs.mean`to define the average vector.
        Notes:
        ------
        - The function extracts the covariance matrix from `obs`.
        - The function converts the model expression into a numerical function and its Jacobian using `expr_to_func()`.
        - The chi-squared minimization is performed using the LevenbergMarquardt class.
        '''
        self.x = x
        self.obs = obs
        self.max_iter = max_iter
        self.tol = tol
        if mean is not None:
            y = mean
        else:
            y = obs.mean
        self.mean = y
        if cov is not None:
            self.cov = cov
        else:
            cov = obs.covariance_matrix()
            self.cov = cov
        if self.W2 == 'diag':
            self.W2 = 1 / numpy.diag(cov)
        elif self.W2 == 'full':
            self.W2 = svd_inv(cov)
        if len(self.W2.shape) == 1:
            self.W2 = numpy.diag(self.W2)
        self.W = svd_sqrt(self.W2)
        #result = scipy.optimize.minimize(chi2, initial_guess, args=(x, y, model, W))
        #result = scipy.optimize.least_squares(chi2_residuals, initial_guess, args=(x, y, model, W),jac=lambda params, *args: numpy.array(jacobian(args[0], *params)).T, method='lm')
        result = LevenbergMarquardt(x, y, self.model, self.jacobian, self.W2, self.initial_guess, self.max_iter, self.tol)
        self.result = result
        if not result.success:
            print('[jackfit.fit] Fit did not converge: ' + result.message)
        else:
            print('[jackfit.fit] Fit did converge: ' + result.message)
        self.best_params = result.x
        self.J = numpy.array(self.jacobian(x, *self.best_params))
        # Compute expected chi^2, covariance matrix of parameters and p-value
        self.chi2obs = self.chi2(self.best_params,x,y)
        self.compute_proj(self.W2,self.J)
        self.chi2exp = self.compute_chi2exp(self.cov,self.W2,self.J)
        self.cov_params = self.compute_cov_params(self.cov,self.W2,self.J)
        self.pvalue = self.compute_pvalue(self.chi2obs,self.cov,self.W2,self.J,num_samples)
        self.params = observable(description='Best parameters of fit')
        self.params.create_from_cov(self.best_params,self.cov_params)
        print(f'[jackfit.fit] chi2obs = {self.chi2obs}')
        print(f'[jackfit.fit] chi2exp = {self.chi2exp[0]} +- {self.chi2exp[1]}')
        print(f'[jackfit.fit] p-value = {self.pvalue[0]} +- {self.pvalue[1]}')
    
    def extrapolate(self, x, plot=False, cov=False, log=False):
        '''
        Extrapolate the fitted model function using the fitted parameters.
        '''
        if self.params is None:
            raise ValueError('[jackfit.extrapolate] Fit has not been performed yet.')
        x = numpy.array(x)
        #return self.pyobs_model(x,self.params) does not correctly propagate errors on estimated parameters
        mean_extr = self.model(x, *self.best_params)
        J = numpy.array(self.jacobian(x, *self.best_params)) # new xs w.r.t those at which we computed self.J
        cov_extr = J.T @ self.cov_params @ J
        extrapolated = observable(description='Extrapolated values')
        extrapolated.create_from_cov(mean_extr, cov_extr)
        if plot:
            self.plot(x,extrapolated,log)
        if cov:
            return extrapolated, cov_extr
        else:
            return extrapolated
    
    def plot(self, x=None, extrapolated=None, log=False):
        if x is None:
            x = numpy.linspace(self.x[0],self.x[-1],100*len(self.x))
        if extrapolated is None:
            extrapolated = self.extrapolate(x, plot=False)
        plt.figure(figsize=(10,6))
        # plt.errorbar(self.x,*(self.obs.error()),color='C0',label=r'Data',fmt='.')
        plt.errorbar(self.x,self.obs.mean,self.obs.err,color='C0',label=r'Data',fmt='.')
        plt_errorbar_fill_color(x,extrapolated.mean,extrapolated.err,color='C1',label=r'Extrapolated')
        plt.legend()
        if isinstance(log,list):
            if log[0]:
                plt.xscale('log')
            if log[1]:
                plt.yscale('log')
        elif log:
            plt.yscale('log')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y(x)$')
        plt.title(r'Fit')
        plt.grid()
        plt.tight_layout()
        plt.draw()
        plt.show()

"""
def jackknife_fit(x, y_samples, model, W, initial_guess):
    '''
    Perform a jackknife resampling to compute parameter errors.
    '''
    n_samples, n_data = y_samples.shape
    
    # Perform fit on central values
    y_central = numpy.mean(y_samples, axis=0)
    best_params, _ = fit_chi2(x, y_central, model, W, initial_guess)
    # Fit for each jackknife sample
    jackknife_params = numpy.zeros((n_samples, len(best_params)))
    for i in range(n_samples):
        y_jackknife = (numpy.sum(y_samples, axis=0) - y_samples[i]) / (n_samples - 1)
        jackknife_params[i], _, _ = fit_chi2(x, y_jackknife, model, inv_cov, initial_guess)
    # Compute jackknife errors
    param_errors = numpy.sqrt((n_samples - 1) * numpy.mean((jackknife_params - best_params) ** 2, axis=0))
    return best_params, param_errors
"""
