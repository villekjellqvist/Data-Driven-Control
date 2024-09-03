## % Imports
import numpy as np
from data_driven_control import RLS_Estimator
import control as ct
import matplotlib.pyplot as plt

    
if __name__ == '__main__':
    num = np.array([1, 1.2])
    den = np.array([1, -1, 0.25])
    sys = ct.TransferFunction(num, den, True)

    num_est = [0,0]
    den_est = [0,0,0]

    RLS = RLS_Estimator(num_est, den_est, 0.95, start_delay=100)

    n = 1000
    u = np.array([(i // 20 +2)%2 for i in range(n)]) + np.concatenate([np.random.normal(0,1,int(n/2)), np.zeros(int(n/2))])
    # u = np.arange(1,n+1)
    y = ct.forced_response(sys, U=u.reshape((1,n))).y.squeeze()
    y_hat = np.zeros_like(y)
    AB=[]
    for i in range(n):
        num, den = RLS.update(y[i], u[i])
        y_hat[i] = RLS.y_hat
        AB.append(np.concat([den.reshape(1,3),num.reshape(1,2)],axis=1))
    AB = np.asarray(AB)

    plt.plot(y, label=r'$y$')
    plt.plot(y_hat, label=r'$\hat{y}$')
    plt.legend()
    plt.show()