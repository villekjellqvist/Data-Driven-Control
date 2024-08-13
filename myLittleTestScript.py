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
A = np.array([[5,4,2,1],[0,1,-1,-1],[-1,-1,3,0],[1,1,-1,2]])
cm.jordan_form(A)