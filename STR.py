# %% Imports
import numpy as np
from data_driven_control import RLS_Estimator, TransferFunction, solve_diophantine, polmul
import control as ct
import matplotlib.pyplot as plt

# %% Update function
Ad = [1, -0.7358, 0.1353] # Desired transfer poles
# Desired transfer fcn for comparison
G = TransferFunction([1, 1.2], [1, -1, 0.25]) # Unknown in reality
Gd = TransferFunction(G.num*(np.sum(Ad)/np.sum(G.num)), Ad)

A0 = [1, -0.5] # Just give it some stable value

# %% Simulations
n = 2000 # Nr of sim steps

# Pre allocating arrays
uc = (#np.concat([np.random.uniform(-0.5,0.5,int(n/2)), np.zeros(int(n/2))])
      + np.array([(i // 100 + 2)%2 for i in range(n)], dtype=float) )# Square wave
u = np.zeros_like(uc)
uv = np.zeros_like(uc)
y = np.zeros_like(uc)
yd = np.zeros_like(uc)
e = np.zeros_like(uc) #+ np.random.uniform(-0.05, 0.05, n)
v = 0

# %%
# Initiating controller and RLS estimator
G = TransferFunction([2, 2.4], [1, -1, 0.25]) # To maintain consistency between runs
Gd.reset()
TR = TransferFunction([1,0],[1,0])
SR = TransferFunction([1,0],[1,0]) 
RLS = RLS_Estimator([1,0.2],[1,0.1,0.3], ff=0.95)
y[0] = G.update(u[0]) + e[0]
yd[0] = Gd.update(uc[0])
for i in range(n-1):
    # Updating RLS estimate
    Bhat, Ahat = RLS.update(y[i], u[i])
    # Updating Controller from RLS
    beta = np.sum(Ad)/np.sum(Bhat)
    T = beta*np.asarray(A0)
    R, S = solve_diophantine(Ahat, Bhat, polmul(Ad, A0), 1, 1)
    TR.tf = (T,R)
    SR.tf = (S,R)
    # Calculating next input and advancing the system
    u[i+1] = TR.update(uc[i]) - SR.update(y[i])
    uv[i+1] = u[i+1] + v
    y[i+1] = G.update(uv[i+1]) + e[i+1]

    # Desired system sim for comparison
    yd[i+1] = Gd.update(uc[i+1])

    # Changing the system sumtimes
    if i == int(n/2):
        G = 2*G

# %% Plots
fig, ax = plt.subplots(2)
ax[0].plot(u, label='$u$')
ax[0].plot(uc, label='$u_c$', linestyle='--')
ax[0].legend()
ax[1].plot(y, label=r'$y=G\ast u$')
ax[1].plot(yd, label=r'$u_c$', linestyle='--')
ax[1].legend()
plt.show()