# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from control.matlab import *
from control import dare


class PreviewControl:
    def __init__(self, z_c):
        print("initializing...")
        self.g = 9.8
        self.z_c = z_c

        self.N_trj = []
        self.f_i_trj = []
        self.p_ref = []
        self.t_trj = []
        self.x_trj = []
        print("initialized.")

    def run(self, t_delta, horizon):
        N = int(horizon/t_delta)
        A = np.array([[1.0, t_delta, t_delta**2/2], [0, 1.0, t_delta], [0, 0, 1.0]])
        b = np.array([[t_delta**3/6], [t_delta**2/2], [t_delta]])
        c = np.array([[1.0, 0, -self.z_c/self.g]])
        Q = np.eye(1)*1.0
        R = np.eye(1)*1.0e-6
        print(A)
        print(b)
        print(c)
        print(c.T.dot(Q).dot(c))
        print(R)

        P, e, K = dare(A, b, c.T.dot(Q).dot(c), R)
        print(P)
        print(K)

        def f_i(i):
            tmp_pA = np.eye(A.shape[0])
            for i in range(i-1):
                tmp_pA = tmp_pA.dot((A-b.dot(K)).T)
            return np.linalg.inv(R+b.T.dot(P).dot(b)).dot(b.T).dot(tmp_pA).dot(c.T).dot(Q)

        for i in range(1, N+1):
            self.N_trj.append(i)
            self.f_i_trj.append(f_i(i)[0])

        print("-----------------------------")

        t_tmp = 0
        # makeing foot step plan. sorry....
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
        while(t_tmp < 7.0):
            self.x_trj.append(x[0][0])

            u = -K.dot(x)
            # print(u)
            for i in range(1, N+1):
                u += np.array([self.f_i_trj[i-1]]).dot(np.array(self.p_ref[cnt+i]))
            x = A.dot(x) + b.dot(u)
            t_tmp += t_delta
            cnt += 1

    def draw(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.N_trj, self.f_i_trj)
        plt.subplot(2, 1, 2)
        plt.plot(self.t_trj, self.p_ref[:len(self.t_trj)])
        plt.plot(self.t_trj, self.x_trj)
        plt.ylim(-0.2, 1.4)
        plt.show()


def test1():
    previewControl = PreviewControl(1.0)
    previewControl.run(0.005, 1.6)
    previewControl.draw()


def main():
    test1()


if __name__ == '__main__':
    main()
