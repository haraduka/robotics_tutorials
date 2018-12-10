# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy


class ModelPredictiveControl:
    def __init__(self, z_c):
        print("initializing...")
        self.g = 9.8
        self.z_c = z_c

        self.p_ref = []
        self.t_trj = []
        self.x_trj = []
        print("initialized.")

    def run(self, horizon, t_delta):
        T = int(horizon/t_delta)
        A = np.array([[1.0, t_delta, t_delta**2/2], [0, 1.0, t_delta], [0, 0, 1.0]])
        b = np.array([[t_delta**3/6], [t_delta**2/2], [t_delta]])
        c = np.array([[1.0, 0, -self.z_c/self.g]])
        Q = np.eye(1)*1.0
        R = np.eye(1)*1.0e-6
        print(A)
        print(b)
        print(c)
        print(Q)
        print(R)

        print("-----------------------------")

        t_tmp = 0
        # foot step plan, sorry...
        while(t_tmp < 10.0):
            if t_tmp < 2.0:
                self.p_ref.append(0.0)
            elif t_tmp < 3.0:
                self.p_ref.append(0.3)
            elif t_tmp < 4.0:
                self.p_ref.append(0.6)
            elif t_tmp < 5.0:
                self.p_ref.append(0.9)
            else:
                self.p_ref.append(1.2)
            if t_tmp < 7.0:
                self.t_trj.append(t_tmp)
            t_tmp += t_delta

        t_tmp = 0
        cnt = 0
        x = np.zeros((3, 1))
        x_variable = cvxpy.Variable((3, T+1))
        u_variable = cvxpy.Variable((1, T))
        while(t_tmp < 7.0):
            self.x_trj.append(x[0][0])

            print(t_tmp)
            x0 = x[:]
            cost = 0.0
            constr = []
            for t in range(T):
                cost += cvxpy.quad_form(c * x_variable[:, t+1] - self.p_ref[cnt+t+1], Q)
                cost += cvxpy.quad_form(u_variable[:, t], R)
                constr.append(x_variable[:, t+1] == A*x_variable[:, t] + (b*u_variable[:, t])[0])
            constr.append(x_variable[:, 0:1] == x0)
            prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
            prob.solve(verbose=False)

            u = np.array([[u_variable.value[0, 0]]])
            print(u)
            print(c.dot(x))

            x = A.dot(x) + b.dot(u)
            t_tmp += t_delta
            cnt += 1

    def draw(self):
        plt.plot(self.t_trj, self.p_ref[:len(self.t_trj)])
        plt.plot(self.t_trj, self.x_trj)
        plt.ylim(-0.2, 1.4)
        plt.show()


def test1():
    modelPredictiveControl = ModelPredictiveControl(1.0)
    modelPredictiveControl.run(1.0, 0.01)
    modelPredictiveControl.draw()


def main():
    test1()


if __name__ == '__main__':
    main()
