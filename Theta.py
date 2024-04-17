import numpy as np

class ThetaObject:
    def __init__(self, thetas):
        self.thetas = thetas
        self.theta_1, self.theta_2, self.theta_3, self.theta_4 = thetas[:4]
        self.theta_5a, self.theta_5b, self.theta_6, self.theta_7 = thetas[4:]
        self.p = self.theta_1.shape[0]

    def __str__(self):
        names = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5a", "theta_5b", "theta_6", "theta_7"]
        r = "\n".join([20*"-" + "\n{}\n{}".format(name, theta) for (name, theta) in zip(names, self.thetas)])
        return r

class RandomThetaObject (ThetaObject):
    def __init__(self, p):
        theta_1 = np.random.rand(p, 1)
        theta_2 = np.random.rand(p, p)
        theta_3 = np.random.rand(p, p)
        theta_4 = np.random.rand(p, 1)
        theta_5a = np.random.rand(p, 1)
        theta_5b = np.random.rand(p, 1)
        theta_6 = np.random.rand(p, p)
        theta_7 = np.random.rand(p, p)
        thetas = [theta_1, theta_2, theta_3, theta_4, theta_5a, theta_5b, theta_6, theta_7]
        super().__init__(thetas)