from numpy import typing as nptyping
import numpy as np
from copy import deepcopy


def assert_vectors_1dim(*coeff) -> list[nptyping.NDArray]:
    vecs: list[nptyping.NDArray] = []
    for c in coeff:
        vecs.append(np.asarray(c, dtype=np.float64).squeeze())
    for v in vecs:
        if (v.ndim > 1):
            raise ValueError("num and den must be 1 dimensional arrays.")
    return vecs


def systemOrder(numerator: nptyping.NDArray, denominator: nptyping.NDArray):
    if (numerator.ndim > 1) or (denominator.ndim > 1):
        raise ValueError("num and den must be 1 dimensional arrays.")

    # Extract model params
    relorder = denominator.size - numerator.size
    if relorder < 0:
        raise ValueError(
            "Numerator order is higher than denominator, system is not causal."
        )
    order = denominator.size - 1

    return (order, relorder)

def polmul(*p):
    """Does polynomial multiplication given lists of polynomial coefficients.

    Returns:
        NDArray of coefficients of result of polynomial multiplication.
    """
    p = assert_vectors_1dim(*p)
    ret = np.polynomial.Polynomial(p[0])
    for i in range(1,len(p)):
        factor = np.polynomial.Polynomial(p[i])
        ret *= factor
    return ret.coef



def solve_diophantine(A: nptyping.ArrayLike, B: nptyping.ArrayLike, Ac: nptyping.ArrayLike, R_order:int, S_order:int):
    """Solves the polynomial Diophantine eq. AR + BS = Ac for R and S given A,B and Ac.
    Coefficients are given in descending order such that:
        R = [1, r1, r2, ... rk]
        S = [s0, s1, ... sl]
        R(x) = [x**k r1x**k-1 ... rk]
        S(x) = [s0x**l s1x**l-1 ... sl]

    Some assumptions are made:
    * R, A and Ac are assumed to be monic. 
    * A and B must not have common factors (i.e. transfer function must be minimal).
    * Left and right sides have the same order.
    * order(S) < order(A), order(B) ≤ order(A)

    Arguments:
        A -- A polynomial coefficients
        B -- B polynomial coefficients
        Ac -- Ac polynomial coefficients
        R_order -- Order of the R polynomial
        S_order -- Order of the S polynomial < order of A

    Raises:
            ValueError: If A or Ac are not monic.

    Returns:
        (R, S): NDarrays of polynomial coefficients
    """
    # Parameter sanity checks
    A, B, Ac = assert_vectors_1dim(A, B, Ac)
    if (not np.isclose(A[0], 1)) or (not np.isclose(Ac[0], 1)):
        raise ValueError('A and Ac must be monic')
    k = R_order
    l = S_order + 1
    # Matrix pre allocating
    rows = Ac.size - 1
    cols = k+l
    A_pad = np.pad(A, (k-1, k-1))
    B_pad = np.pad(B, (l-1, l-1))
    Syl = np.zeros((rows, cols))
    # Creating Sylvester matrix
    for r in range(rows):
        Syl[r, :k] += np.flip(A_pad[r:r+k])
        Syl[r, k:] += np.flip(B_pad[r:r+l])
    
    # Right hand side matrix
    RM = Ac[1:].reshape((rows,1))
    RM[:A.size-1] -= A[1:].reshape((A.size-1,1))

    # Solving for R and S
    rs = (np.linalg.pinv(Syl)@RM).flatten()
    R = np.concat([[1], rs[:k]])
    S = rs[k:]

    return (R,S)

## % RLS estimator class
class RLS_Estimator:
    def __init__(
        self,
        numerator0: nptyping.ArrayLike,
        denominator0: nptyping.ArrayLike,
        ff: float,
        P0: nptyping.ArrayLike = None,
        start_delay: int = 0,
    ):
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
        self.num, self.den = assert_vectors_1dim(numerator0, denominator0)

        # Extract model params
        self.order, self.relorder = systemOrder(self.num, self.den)
        self.nr_params = 2 * self.order - self.relorder + 1

        # Init RLS algo
        self.P = np.eye(self.nr_params) * 1000 if P0 is None else np.asarray(P0)
        if (self.P.shape[0] != self.nr_params) or (self.P.shape[1] != self.nr_params):
            raise ValueError(
                "P0 must be a square matrix of shape nxn; n = len(num) + len(den)"
            )
        self.K = np.zeros((self.nr_params, 1))

        if ff > 1 or ff < 0:
            raise ValueError("Forgetting factor ff must be 0<ff≤1")
        self.ff = ff

        self.phi = np.full((self.nr_params, 1), np.nan)
        self.u = np.full((self.order + 1, 1), np.nan)
        self.y = np.full((self.order + 1, 1), np.nan)

        # Init inital estimate if required.
        self.startEstimating = False
        self.start_delay = start_delay
        self.LS_iter = 0
        if self.start_delay > 0:
            if self.start_delay < self.order + 1:
                self.start_delay = 0
            else:
                self.uLS = np.zeros((self.start_delay, 1))
                self.yLS = np.zeros((self.start_delay, 1))

    @property
    def theta(self):
        return np.concatenate([self.den[1:], self.num], axis=0).reshape(
            (self.nr_params, 1)
        )

    @theta.setter
    def theta(self, theta_new: nptyping.ArrayLike):
        theta_new = np.asanyarray(theta_new).squeeze()
        if (theta_new.size != self.nr_params) or (theta_new.ndim != 1):
            raise ValueError("Theta must be a vector with size len(num) + len(den).")
        self.den[1:] = theta_new[: self.order]
        self.den[0] = 1
        self.num = theta_new[self.order :]

    @property
    def y_hat(self):
        return (self.phi.T @ self.theta).item() if self.startEstimating else np.nan

    def __intial_estimate_wait(self, y, u):
        self.startEstimating = (
            np.isnan(self.phi).sum() == 0 and self.LS_iter >= self.start_delay
        )
        if not self.startEstimating:
            if self.start_delay > 0:
                self.uLS[self.LS_iter] = u
                self.yLS[self.LS_iter] = y
                if self.LS_iter == self.start_delay - 1:
                    self.__estimate_inital_guess()
                self.LS_iter += 1
        return not self.startEstimating

    def __estimate_inital_guess(self):
        numerator_params = self.order - self.relorder + 1
        Hu = np.concat(
            [self.uLS[i : -(numerator_params - i)] for i in range(0, numerator_params)],
            axis=1,
        )
        Hy = np.concat(
            [-self.yLS[i : -(self.order - i)] for i in range(0, self.order)], axis=1
        )
        H = np.concat([np.fliplr(Hy), np.fliplr(Hu)], axis=1)
        theta_est = np.linalg.pinv(H) @ self.yLS[self.order :]
        self.theta = theta_est
        self.P = np.linalg.inv(H.T @ H)

    def __update_phi(self, y: float, u: float):
        self.u[1:] = self.u[:-1]
        self.u[0] = u
        self.y[1:] = self.y[:-1]
        self.y[0] = y

        self.phi[: self.order] = -self.y[1:]
        self.phi[self.order :] = self.u[self.relorder :]

    def update(s, y: float, u: float):
        s.__update_phi(y, u)
        # Don't start before given delay.
        if s.__intial_estimate_wait(y, u):
            return (s.num, s.den)

        # RMS
        s.K = s.P @ s.phi / (s.ff + s.phi.T @ s.P @ s.phi).item()
        eps = y - s.y_hat
        s.theta = s.theta + s.K * eps

        s.P = (np.eye(s.nr_params) - s.K @ s.phi.T) @ s.P / s.ff
        return (s.num, s.den)


class TransferFunction:
    def __init__(self, numerator, denominator):
        self.num, self.den = assert_vectors_1dim(numerator, denominator)
        self.order, self.relorder = systemOrder(self.num, self.den)
        self.nr_params = 2 * self.order - self.relorder + 1

        # Zero initial conditions
        self.phi = np.zeros((self.nr_params, 1), dtype=np.float64)
        self.u = np.zeros((self.order + 1, 1), dtype=np.float64)
        self.y = np.zeros((self.order, 1), dtype=np.float64)

    def __call__(self, z: float):
        numorder = self.order - self.relorder

        num = 0
        for i in range(numorder + 1):
            num += self.num[i] * z ** (numorder - i)

        den = 0
        for i in range(self.order + 1):
            den += self.den[i] * z ** (self.order - i)
        return num / den

    @property
    def tf(self):
        return (self.num, self.den)

    @tf.setter
    def tf(self, numden):
        num, den = assert_vectors_1dim(numden[0], numden[1])
        self.num = num
        self.den = den

    @property
    def theta(self):
        return np.concatenate([self.den[1:], self.num], axis=0).reshape(
            (self.nr_params, 1)
        )

    def __mul__(self, x: float):
        ret = deepcopy(self)
        ret.tf = (x * self.num, self.den)
        return ret

    __rmul__ = __mul__

    def __update_u(self, u: float):
        self.u[1:] = self.u[:-1]
        self.u[0] = u
        self.phi[self.order :] = self.u[self.relorder :]

    def __update_y(self, y: float):
        self.y[1:] = self.y[:-1]
        self.y[0] = y
        self.phi[: self.order] = -self.y

    def reset(self):
        self.phi[:] = 0
        self.u[:] = 0
        self.y[:] = 0

    def update(self, u: float):
        self.__update_u(u)
        y = (self.phi.T @ self.theta).item()
        self.__update_y(y)
        return y

    def __repr__(self) -> str:
        format = {"float": lambda x: f"{x:.03f}"}
        return (
            f"{np.array2string(self.num, formatter=format)}"
            + f"/{np.array2string(self.den, formatter=format)}"
        )

    def _repr_latex_(self):
        numorder = self.order - self.relorder

        z = lambda p: f"z^{p} + " if p > 1 else f"z + " if p == 1 else ""
        num = ""
        for i in range(numorder + 1):
            num += f"{self.num[i]:.03f}" + f"{z(numorder - i)}"
        den = ""
        for i in range(self.order + 1):
            den += f"{self.den[i]:.03f}" + f"{z(self.order - i)}"
        return rf"$\frac{{ {num} }}{{ {den} }}$"
