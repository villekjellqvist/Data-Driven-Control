from numpy import typing as nptyping
import numpy as np

## % RLS estimator class
class RLS_Estimator():
    def __init__(self, numerator0:nptyping.ArrayLike, denominator0:nptyping.ArrayLike, ff:float, P0:nptyping.ArrayLike=None, start_delay:int = 0):
        """Initiates a Recursive Least Squares Estimator for a SISO transfer function.

        Arguments:
            numerator0 -- Initial guess for numerator
            denominator0 -- Initial guess for denominator
            ff -- Forgetting Factor in the range(0,1]

        Keyword Arguments:
            P0 -- Initial parameter covariance guess. Must be a nxn matrix where n = size(num) + size(den) - 1
            start_delay -- 
                - If smaller than the system order: Will start estimating as soon as there are
                    at least as many samples as the order of the system. 
                    This uses the inital estimates of the numerator and denominator
                    provided during initialization, and may cause divergence for bad guesses.
                - If greater than system order: Will wait for this many samples, do a least squares fit
                    and then use that as a start guess for the inital estimates. May diverge if first
                    samples are not sufficiently exciting.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        # Init model
        self.num = np.asarray(numerator0, dtype=np.float64).squeeze()
        self.den = np.asarray(denominator0, dtype=np.float64).squeeze()
        if (self.num.ndim > 1) or (self.den.ndim > 1):
            raise ValueError("num and den must be 1 dimensional arrays.")
        
        # Extract model params
        self.relorder = self.den.size - self.num.size
        if self.relorder < 0:
            raise ValueError("Numerator order is higher than denominator, system is not causal.")
        
        self.numerator_params = self.num.size
        self.order = self.den.size - 1
        self.nr_params = self.order + self.numerator_params

        # Init RLS algo
        self.P = np.eye(self.nr_params)*1000 if P0 is None else np.asarray(P0)
        if (self.P.shape[0] != self.nr_params) or (self.P.shape[1] != self.nr_params):
            raise ValueError("P0 must be a square matrix of shape nxn; n = len(num) + len(den)")
        self.K = np.zeros((self.nr_params, 1))

        if (ff > 1 or ff<0):
            raise ValueError("Forgetting factor ff must be 0<ff≤1")
        self.ff = ff

        self.phi = np.full((self.nr_params, 1), np.nan)
        self.u = np.full((self.numerator_params+self.relorder, 1), np.nan)
        self.y = np.full((self.order+1, 1), np.nan)

        # Init inital estimate if required.
        if start_delay is None:
            start_delay = 4*self.order
        self.start_delay = start_delay
        self.iter = 0
        if self.start_delay > 0:
            if self.start_delay < self.order + 1:
                self.start_delay = 0
            else:
                self.uLS = np.zeros((self.start_delay,1))
                self.yLS = np.zeros((self.start_delay,1))


    @property
    def theta(self):
        return np.concatenate([-self.den[1:], self.num], axis=0).reshape((self.nr_params, 1))
    
    @theta.setter
    def theta(self, theta_new:nptyping.ArrayLike):
        theta_new = np.asanyarray(theta_new).squeeze()
        if (theta_new.size != self.nr_params) or (theta_new.ndim != 1):
            raise ValueError("Theta must be a vector with size len(num) + len(den).")
        self.den[1:] = -theta_new[:self.order]
        self.den[0] = 1
        self.num = theta_new[self.order:]

    @property
    def y_hat(self):
        return (self.phi.T@self.theta)[0,0]
    
    def __estimate_inital_guess(self):
        Hu = np.concat([self.uLS[i:-(self.numerator_params-i)] for i in range(0,self.numerator_params)],axis=1)
        Hy = np.concat([self.yLS[i:-(self.order-i)] for i in range(0,self.order)],axis=1)
        H = np.concat([np.fliplr(Hy), np.fliplr(Hu)], axis=1)
        theta_est = np.linalg.pinv(H)@self.yLS[self.order:]
        self.theta = theta_est
        self.P = np.linalg.inv(H.T@H)
    
    def __update_phi(self, y:float, u:float):
        self.u[1:] = self.u[:-1]
        self.u[0] = u
        self.y[1:] = self.y[:-1]
        self.y[0] = y

        self.phi[:self.order] = self.y[1:]
        self.phi[self.order:] = self.u[self.relorder:]
    
    def update(s, y:float, u:float):
        s.__update_phi(y,u)
        # Don't start before given delay.
        if np.isnan(s.phi).sum() > 0 or s.iter < s.start_delay:
            if s.start_delay > 0:
                s.uLS[s.iter] = u
                s.yLS[s.iter] = y
                if s.iter == s.start_delay-1:
                    s.__estimate_inital_guess()
                s.iter += 1

            return (s.num, s.den)
        
        # RMS
        s.K = s.P@s.phi / (s.ff + s.phi.T@s.P@s.phi)[0,0]
        eps = y - s.y_hat
        s.theta = s.theta + s.K*eps

        s.P = (np.eye(s.nr_params) - s.K@s.phi.T)@s.P/s.ff
        return (s.num, s.den)