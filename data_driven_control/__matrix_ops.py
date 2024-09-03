import numpy as np
import sympy as sym
from data_driven_control.common import MatrixLike
from scipy.linalg import sqrtm

def check_state_space_dimensions(A:MatrixLike, B:MatrixLike=None, C:MatrixLike=None, D:MatrixLike=None): # type: ignore
    """Checks that state space matrices are of proper dimension(s) relative to each other.

    Arguments:
        A -- nxn system matrix

    Keyword Arguments:
        B -- nxp input matrix (default: {None})
        C -- qxn output matrix (default: {None})
        D -- qxp feedthrough matrix (default: {None})

    Raises:
        ValueError: If matrix dimensions are not of proper dimension(s).
    """
    dims = lambda M: f"{M.shape[0]}x{M.shape[1]}"
    ops = matrix_ops.check_sympy_or_numpy(A,B,C,D)
        
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, but has dimensions {dims(A)}.")

    if B is not None:
        if B.shape[0] != A.shape[0]:
            raise ValueError(f"B must have the same number of rows (or states) as A, "+
                            f"but A is {dims(A)} and B is {dims(B)}.")
    
    if C is not None:
        if C.shape[1] != A.shape[0]:
            raise ValueError(f"C must have the same number of columns as rows in A, but "+
                            f"C is {dims(C)} and A is {dims(A)}.")
    
    if D is not None:
        if D.shape[1] != B.shape[1]:
            raise ValueError(f"D must have the same number of columns as B, but "+
                         f"D is {dims(D)} and B is {dims(B)}.")
        if D.shape[0] != C.shape[0]:
            raise ValueError(f"D must have the same number of rows (or outputs) as C, but "+
                         f"D is {dims(D)} and C is {dims(C)}.")
        


class matrix_ops():
    def __init__(self, sympyOrNumpy:str):
        self.__notInit = ("Sympy or numpy state not initialized. This is likely "+
            "because __check_sympy_or_numpy was not called properly before " +
            "using internal matrix ops in symbolic_control.")
        self.__sympyOrNumpy = sympyOrNumpy

    @classmethod
    def check_sympy_or_numpy(cls, *Mat:MatrixLike):
        """Checks if given Matrices are sympy or numpy, and sets
        function aliases to the correct operation.

        Raises:
            ValueError: If one (or more) matrices are not sympy or numpy.
            ValueError: If matrices are not of same type.
        """
        i = 0
        for M in Mat:
            if M is None:
                continue
            if i == 0:
                MType = type(M)
            if isinstance(M, sym.Matrix):
                sympyOrNumpy = 'Sympy'
            elif isinstance(M, np.ndarray):
                sympyOrNumpy = 'Numpy'
            else:
                sympyOrNumpy = 'None'
                raise ValueError(f"Matrices must be either sympy matrices or numpy NDArrays, but are of type {type(M)}.")
            if type(M) != MType:
                sympyOrNumpy = 'None'
                raise ValueError(f"All matrices must be of same type, but there are types {type(M)} and {MType}.")
            MType = type(M)
            i += 1
        return cls(sympyOrNumpy)

    def conc(self, L:list[MatrixLike], axis:int):
        if self.__sympyOrNumpy == 'Sympy':
            return sym.Matrix(L) if axis==0 else sym.Matrix([L])
        elif self.__sympyOrNumpy == 'Numpy':
            return np.concatenate(L,axis)
        else:
            raise RuntimeError(self.__notInit)
    
    
    def zeros(self, m:int,n:int):
        if self.__sympyOrNumpy == 'Sympy':
            return sym.zeros(m,n)
        elif self.__sympyOrNumpy == 'Numpy':
            return np.zeros((m,n))
        else:
            raise RuntimeError(self.__notInit)
        
    def matrix_power(self, M:MatrixLike,n:int):
        if self.__sympyOrNumpy == 'Sympy':
            return M**n
        elif self.__sympyOrNumpy == 'Numpy':
            if(n==1/2):
                return sqrtm(M)
            return np.linalg.matrix_power(M,n)
        else:
            raise RuntimeError(self.__notInit)
        
    def as_row_vector(self, v:MatrixLike):
        if self.__sympyOrNumpy == 'Sympy':
            if v.shape[0] > 1 and v.shape[1] > 1:
                raise ValueError("v must be a one dimensional vector")
            size = v.shape[0]*v.shape[1]
            return v.reshape(size,1)
        elif self.__sympyOrNumpy == 'Numpy':
            ret = np.squeeze(v)
            if ret.ndim > 1:
                raise ValueError("v must be a one dimensional vector")
            ret = np.reshape(ret, (ret.size,1))
        else:
            raise RuntimeError(self.__notInit)
        return ret