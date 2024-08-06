import sympy as sym
import numpy as np
import numpy.typing as nptyping
from IPython.display import display

def printMatrix(Mstr:str, M:nptyping.NDArray):
    Mprint = sym.Matrix(M)
    rows = M.shape[0]
    try:
        cols = M.shape[1]
    except IndexError:
        cols = 1
    Msym = sym.MatrixSymbol(Mstr, rows, cols)
    display(sym.Eq(Msym, Mprint),)

if __name__ == '__main__':
    A = np.array([[0,1],[2,3]])
    B = np.array([4,5,6,7,8])
    Th = np.array([[9,10,11,12,13,14]])
    printMatrix('A',A)
    printMatrix('B',B)
    printMatrix(r'Theta',Th)