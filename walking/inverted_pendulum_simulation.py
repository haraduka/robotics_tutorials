# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class InvertedPendulumSimulation:
    def __init__(self, M, initial_angle, height, kind):
        self.g = 9.8
        self.M = M
        self.kind = kind

        self.p_trj = [initial_angle]  # trajectory
        self.pp_trj = [0.0]  # rad
        self.r_trj = [height]
        self.rr_trj = [0.0]
        self.t_trj = [0.0]

    def f_func(self, p, pp, r, rr):
        if self.kind == 0:
            return 0
        elif self.kind == 1:
            return self.M*self.g/np.cos(p)
        elif self.kind == 2:
            return self.M*self.g*np.cos(p)-self.M*r*pp**2
        elif self.kind == 3:
            return self.M*self.g

    def tau_func(self, p, pp, r, rr):
        return 0.0

    # (θ, dθ, r, dr, τ, f)
    def f0(self, p, pp, r, rr):  # dp =
        return pp

    def f1(self, p, pp, r, rr):  # dr =
        return rr

    def f2(self, p, pp, r, rr, tau):  # dpp =
        return (self.tau_func(p, pp, r, rr)/self.M - 2.0*r*rr*pp + self.g*r*np.sin(p))/(r*r)

    def f3(self, p, pp, r, rr, f):  # drr =
        return r*pp**2 - self.g*np.cos(p) + self.f_func(p, pp, r, rr)/self.M

    # h: time step of simulation
    def run(self, sec_, h_=0.002):
        # initialize
        p = self.p_trj[0]
        pp = self.pp_trj[0]
        r = self.r_trj[0]
        rr = self.rr_trj[0]
        t = self.t_trj[0]

        h = h_

        # main loop
        while len(self.t_trj) <= sec_/h:
            now_p = p + h*self.f0(p, pp, r, rr)
            now_r = r + h*self.f1(p, pp, r, rr)
            now_pp = pp + h*self.f2(p, pp, r, rr, 0.)
            now_rr = rr + h*self.f3(p, pp, r, rr, 0)
            t = t + h

            p = now_p
            r = now_r
            pp = now_pp
            rr = now_rr

            self.p_trj.append(p)
            self.r_trj.append(r)
            self.pp_trj.append(pp)
            self.rr_trj.append(rr)
            self.t_trj.append(t)

            # runge kutta is also good.
            # k1 = h*f(v)
            # k2 = h*f(v+0.5*k1)
            # k3 = h*f(v+0.5*k2)
            # k4 = h*f(v+k3)
            # v = v + (k1+2*k2+2*k3+k4)/6
            # x = x - v*h
            # t = t + h
            # vtrj = np.append(vtrj,v)
            # xtrj = np.append(xtrj,x)
            # array_t = np.append(array_t,t)

    def draw(self):
        # draw
        x_trj = map(lambda a, b: a*np.sin(b), self.r_trj, self.p_trj)
        y_trj = map(lambda a, b: a*np.cos(b), self.r_trj, self.p_trj)
        plt.plot(x_trj, y_trj)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.show()


def test1():
    invertedPendulumSimulation = InvertedPendulumSimulation(54.0, 0.2, 0.8, 0)
    invertedPendulumSimulation.run(1.0)
    invertedPendulumSimulation.draw()


def test2():
    invertedPendulumSimulation = InvertedPendulumSimulation(54.0, 0.2, 0.8, 1)
    invertedPendulumSimulation.run(1.0)
    invertedPendulumSimulation.draw()


def test3():
    invertedPendulumSimulation = InvertedPendulumSimulation(54.0, 0.2, 0.8, 2)
    invertedPendulumSimulation.run(1.0)
    invertedPendulumSimulation.draw()


def test4():
    invertedPendulumSimulation = InvertedPendulumSimulation(54.0, 0.2, 0.8, 3)
    invertedPendulumSimulation.run(1.0)
    invertedPendulumSimulation.draw()


def main():
    test1()
    test2()
    test3()
    test4()


if __name__ == '__main__':
    main()
