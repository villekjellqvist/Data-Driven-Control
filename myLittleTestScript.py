# %% Imports
import sympy as sym
import numpy as np
import control_matrices as cm
import control as ct
import matplotlib.pyplot as plt
import matrepr
from PIL import Image
import io

# %% System
A = np.array([[0,1,0,0],
              [-0.5,-0.4,0,-0.5],
              [0,0,0.2,-1],
              [0,0,0,-0.8]]
              )

B = np.array(
    [[0,0],
     [1,0],
     [0,0],
     [0,1]]
)
C = np.array(
    [[1,0,0,0],
     [0,0,2,1]]
    )
D = np.array([[0.0, 0.0], [0.0, 0.0]])

sys = ct.ss(A,B,C,0)
# %%
MP = cm.compute_markov_parameters(A,B,C,D, N=10)
matrepr.mdisplay(MP, title='Markov Parameters')
H = MP.com

