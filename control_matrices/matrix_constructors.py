import numpy as np
from control_matrices.common import MatrixLike
from control_matrices.__matrix_ops import matrix_ops, check_state_space_dimensions
    

def interleave_vectors(*v:MatrixLike):
    ops = matrix_ops.check_sympy_or_numpy(*v)
    nargs = len(v)
    vecs = []
    vshape = v[0].shape
    vecs.append( ops.as_row_vector(v[0]))
    retsize = vecs[0].shape[0]
    for i in range(1,nargs):
        if v[i].shape != vshape:
            raise ValueError("All vectors must have same size, but "+
                             f"vector {i-1} has shape {v[i-1].shape} and "+
                             f"vector {i} has shape {v[i].shape}.")
        vecs.append(ops.as_row_vector(v[i]))
        retsize += vecs[i].shape[0]
        
    ret = ops.zeros(retsize,1)
    for i in range(0,nargs):
        ret[i::nargs] = vecs[i]
    return ret


def make_input_Toeplitz_matrix(u:MatrixLike, p):
    """Generates a Toeplitz matrix inputs from an input
    vector u in the form:
    
    | u[0]  u[1]  ...  u[N-1] |
    |  0    u[0]  ...  u[N-2] |
    |  0     0    ...  u[N-3] |
    | ...   ...   ...   ...   |
    |  0     0    ...   u[0]  |

    Arguments:
        u -- A pNx1 block vector, where N is number of input samples.

    Keyword Arguments:
        p -- Number of rows per block (default: {1})

    Raises:
        ValueError: If number of block rows does not fit evenly
            in u.

    Returns:
        An input vector Toeplitz matrix.
    """
    ops = matrix_ops.check_sympy_or_numpy(u)
    u = ops.as_row_vector(u)
    rows = u.shape[0]
    N = int(rows/p)
    if (rows%p != 0):
        raise ValueError(f"v must be a pNx1 block vector, but given block size p={p} "+
                         f"doesn't fit in v with dimensions {u.shape}.")

    U = ops.zeros(p*N,N)
    for c in range(0, N):
        U[0:(c+1)*p,c] = np.flip(np.squeeze(u[0:(c+1)*p]))
    return U


def compute_markov_parameters(A:MatrixLike, B:MatrixLike, C:MatrixLike, D:MatrixLike=None, N:int=0): # type: ignore
    """Generates a matrix of Markov parameters from a
    state space system in the form:
    
    | D  CB  CAB  ...  CA**(i-2)B |

    Arguments:
        A -- nxn system matrix
        B -- nxp input matrix
        C -- qxn output matrix

    Keyword Arguments:
        D -- qxp feedthrough matrix (default: |0|)
        t -- Number of block columns or timesteps in matrix (inferred default: {n})

    Returns:
        A matrix M of markov parameters with dim(M) = qxpN
    """
    ops = matrix_ops.check_sympy_or_numpy(A,B,C,D)
    check_state_space_dimensions(A,B,C,D)

    n = A.shape[0]
    p = B.shape[1]
    q = C.shape[0]
    if N == 0:
        N = n
    if D is None:
        D = ops.zeros(q, p)
        
    
    rows = q
    cols = p*(N)

    M = ops.zeros(rows, cols)

    for c in range(-1,N-1):
        M[:,p*(c+1):p*(c+1)+p] = D if c == -1 else C@np.linalg.matrix_power(A,c)@B
    return M

def make_observability_matrix(A:MatrixLike, C:MatrixLike, t:int=None):# type: ignore
    """Creates an observability block matrix from an A,C matrix pair
    in the form:
        |     C      |
        |    CA      |
        |   CA**2    |
        |    ...     |
        | CA**(t-1)  |

    Arguments:
        A -- nxn system matrix
        C -- qxn output matrix

    Keyword Arguments:
        t -- Number of block rows or timesteps in observability
            matrix (inferred default: {n})

    Returns:
        An observability matrix O with dim(O) = q*txn
    """
    check_state_space_dimensions(A,C=C)
    ops = matrix_ops.check_sympy_or_numpy(A,C)
    n = A.shape[1]
    if t is None:
        t = n

    R = []
    for i in range(0,t):
        elem = C@ops.matrix_power(A,i)
        R.append(elem)
    O = ops.conc(R, axis=0)
    return O

def make_controllability_matrix(A:MatrixLike, B:MatrixLike, t:int=None):# type: ignore
    """Creates a controllability block matrix from an A,B matrix pair in the
    form:

    | B AB A(**2)B ... (A**t-1)B |


    Arguments:
        A -- nxn system matrix
        B -- nxp input matrix

    Keyword Arguments:
        t -- Number of block columns or timesteps in controllability matrix (inferred default: {n})

    Returns:
        Controllability matrix S with dim(S) = nxp*n
    """
    check_state_space_dimensions(A,B)
    ops = matrix_ops.check_sympy_or_numpy(A,B)

    n = A.shape[1]
    if t is None:
        t = n

    R = []
    for i in range(0,t):
        elem = (np.linalg.matrix_power(A, i))@B
        R.append(elem)
    S = ops.conc(R, axis=1)
    return S

def make_Hankel_matrix(v:MatrixLike, block_rows:int, c:int=1,r:int=1):
    """Makes a block Hankel matrix from a column block vector.
       A block vector v=(B1 B2 B3 ... Bn) will return a Hankel
       matrix of the form

       |  B1  B2  B3 ... Bn-t-1 |
       |  B2  B3   ...    ...   |
       | ...          ...       |
       | Bt-1 Bt Bt+1 ...  Bn   |


    Arguments:
        v -- input block vector
        block_rows -- number of block rows t

    Keyword Arguments:
        c -- number of columns in blocks (default: {1})
        r -- number of rows in blocks (default: {1})

    Raises:
        ValueError: If shape of v doesn't match p or q
        ValueError: If t is larger than the number of block rows.

    Returns:
        H -- Block Hankel Matrix
    """
    ops = matrix_ops.check_sympy_or_numpy(v)
    try:
        cols = v.shape[1]
    except IndexError:
        v = v.reshape((1,v.shape[0]))
        cols = v.shape[1]
    rows = v.shape[0]
    if (rows != r) or (cols%c != 0):
        raise ValueError(f"v must be a column block vector, but given block size qxp=({r},{c}) "+
                         f"doesn't fit in v with dimensions {v.shape}.")
    block_cols = int(cols/c)
    if block_rows > block_cols:
        raise ValueError(f"block_rows can't be larger than the number of block " +
                         f"columns in input vector. block_rows={block_rows} and number of block columns={block_cols}")
    T = block_cols - block_rows
    H = ops.zeros(block_rows*r,(T+1)*c)

    for i in range(block_rows):
        H[i*r:(i+1)*r,:] = v[:,i*c:(i+T+1)*c]
    return H

def jordan_form(M:MatrixLike):
    ops = matrix_ops.check_sympy_or_numpy(M)
    evals, evecs = np.linalg.eig(M)
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:,idx]
    dups = []
    for i in range(1,evals.size):
        if np.isclose(evals[i-1],evals[i], atol=1e-12):
            evals[i] = evals[i-1]
            evecs[:,i] = evecs[:,i-1]
            dups.append(i-1)
    J = np.diag(evals)
    for i in dups:
        J[i,i+1] = 1
    pass

