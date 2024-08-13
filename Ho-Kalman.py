# %% Imports
import sympy as sym
import numpy as np
import control_matrices as cm
import control as ct
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from matrepr import mdisplay

# %% System and sim parameters
A = np.array([[0.5,0,0,0],[0.5,-0.5,0.5,0],[0,0,-0.8,0],[0,0,0,-0.3]])
B = np.array([[0.5,0,-1,-0.5]]).T
C = np.array([[0,1,0,-1]])
D = np.array([[0]])

cm.check_state_space_dimensions(A, B, C ,D)
cm.make_observability_matrix(A,C)
cm.make_controllability_matrix(A,B)
n = A.shape[0]
p = B.shape[1]
q = C.shape[0]

steps = 10000

# %% Signals and sim
u = np.zeros(steps)
u[0]=1

x0 = np.zeros((n,1))
x = np.zeros((n,steps))
x[:,0] = x0.T
y = np.zeros(steps)

for i in range (1, steps):
    x[:,i] = A@x[:,i-1] + (B*u[i-1]).T
    y[i-1] = (C@x[:,i-1])[0] + D*u[i-1]
y[-1] = (C@x[:,-1])[0] + D*u[-1]

#%% Markov parameter from input-output
u2 = np.random.uniform(-50,50,steps)
y2 = np.zeros(steps)

for i in range (1, steps):
    x[:,i] = A@x[:,i-1] + (B*u2[i-1]).T
    y2[i-1] = (C@x[:,i-1])[0] + D*u2[i-1]
y2[-1] = (C@x[:,-1])[0] + D*u2[-1]

Y = y2.reshape((1,y.size))
U = cm.make_input_Toeplitz_matrix(u2.reshape(u2.size,1),1)
MP = Y@np.linalg.pinv(U)
MP_True = cm.compute_markov_parameters(A,B,C, N= 10)
H0_2 = cm.make_Hankel_matrix(MP[:,1:-1], int(MP.size/2))
U1, s1 ,Vh1 = np.linalg.svd(H0_2, full_matrices=False)
mdisplay(MP)
mdisplay(MP_True)
mdisplay(s1)

# %% Hankel matrices
H0 = cm.make_Hankel_matrix(y[1:-1], int(y.size/2))
H1 = cm.make_Hankel_matrix(y[2:], int(y.size/2))
# Extracting observability and controllability matrix from SVD of H0
U, s ,Vh = np.linalg.svd(H0, full_matrices=False)
print('Singular Values of H:\n',s)
s[np.isclose(np.zeros(s.shape),s,)] = 0
S = np.diag(s)

O = U@sqrtm(S, 1/2)
O = O[:,~np.all(O == 0, axis=0)]
Cr = sqrtm(S, 1/2)@Vh
Cr = Cr[~np.all(Cr==0, axis=1),:]

# %% Extracting Matrices
nhat = s.size
Ahat = np.linalg.pinv(O)@H1@np.linalg.pinv(Cr)
Bhat = Cr[0:nhat,0:p]
Chat = O[0:q,0:nhat]
Dhat = y[0:q]

# %% estimated sims
xhat = np.zeros((n,steps))
xhat[:,0] = x0.T
yhat = np.zeros(steps)

for i in range (1, steps):
    xhat[:,i] = Ahat@xhat[:,i-1] + (Bhat*u[i-1]).T
    yhat[i-1] = (Chat@xhat[:,i-1])[0] + Dhat*u[i-1]
yhat[-1] = (C@x[:,-1])[0] + Dhat*u[-1]
# %% Lil' plot
plt.plot(u, label='$u$', color='k')
plt.plot(y, label='$y$', color='r', linestyle='--')
plt.plot(yhat, label=r'$\hat{y}$', color='b', linestyle=':')
plt.legend()
plt.show()
# %%
