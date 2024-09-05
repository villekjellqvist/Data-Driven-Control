## % Imports
import numpy as np
from data_driven_control import RLS_Estimator, TransferFunction
import control as ct
import matplotlib.pyplot as plt

    
if __name__ == '__main__':
    num = np.array([1, 1.2])
    den = np.array([1, -1, 0.25])
    sys = ct.TransferFunction(num, den, True)

    num_est = [0,0]
    den_est = [0,0,0]

    RLS = RLS_Estimator(num_est, den_est, 0.95, start_delay=0)

    n = 1000
    u = np.array([(i // 20 +2)%2 for i in range(n)]) + np.concatenate([np.random.normal(0,1,int(n/2)), np.zeros(int(n/2))])
    # u = np.arange(1,n+1)
    Ho = ct.TransferFunction([1, 1.2], [1, -1, 0.25], True, name=r'$H_o$')
    Ho2 = TransferFunction([1, 1.2], [1, -1, 0.25])

    n = 200
    u = np.array([(i // 20 +2)%2 for i in range(n)]) + np.concatenate([np.random.normal(0,1,int(n/2)), np.zeros(int(n/2))])
    #u = np.arange(1,n+1)

    y1 = ct.forced_response(Ho, U=u).y.squeeze()
    y2 = np.zeros_like(y1)

    for i in range(n):
        y2[i] = Ho2.step(u[i])

    plt.plot(y1, label='$y_1$')
    plt.plot(y2[1:], label='$y_2$')
    plt.legend()
    plt.show();