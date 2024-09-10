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
    HoC = ct.TransferFunction([1, 1.2], [1, -1, 0.25], dt=True)
    Ho = TransferFunction([1, 1.2], [1, -1, 0.25])

    n = 200
    u = np.array([(i // 20 +2)%2 for i in range(n)]) + np.concatenate([np.random.normal(0,1,int(n/2)), np.zeros(int(n/2))])
    #u = np.arange(1,n+1)

    y1 = np.zeros(n)
    y2 = np.zeros_like(y1)
    y3 = ct.forced_response(HoC, U=u).y.squeeze()

    for i in range(n):
        y1[i] = Ho.step(u[i])
        Bhat, Ahat = RLS.update(y1[i], u[i])
        y2[i] = RLS.y_hat
        print(f'Bhat: {Bhat}; Ahat: {Ahat}')

    plt.plot(y1, label='$y_1$')
    plt.plot(y2, label='$\hat{y}$', linestyle='--')
    plt.legend()
    plt.show();