import numpy as np

class Theta:
    def __init__(self, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7):
        self.p = theta_1.shape[0]
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.theta_4 = theta_4
        self.theta_5 = theta_5
        self.theta_6 = theta_6
        self.theta_7 = theta_7

class RandomTheta (Theta):
    def __init__(self, p):
        theta_1 = np.random.rand(p, 1)
        theta_2 = np.random.rand(p, p)
        theta_3 = np.random.rand(p, p)
        theta_4 = np.random.rand(p, 1)
        theta_5 = np.random.rand(2 * p, 1)
        theta_6 = np.random.rand(p, p)
        theta_7 = np.random.rand(p, p)
        super().__init__(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7)