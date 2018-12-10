# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LIPModelSimulation:
    def __init__(self, foot_height, foot_width, z_c):
        print("initializing...")
        self.g = 9.8
        self.height = foot_height
        self.width = foot_width
        self.z_c = z_c

        self.x_trj = []
        self.vx_trj = []
        self.y_trj = []
        self.vy_trj = []
        self.t_trj = []

        self.p_x = [0.0]
        self.p_y = [-0.1]
        self.p_x_star = [self.p_x[0]]
        self.p_y_star = [self.p_y[0]]

        print("initialized.")

    def run(self, foot_step, T_sup, h, a=10.0, b=1.0):
        s_x = np.array(foot_step)[:, 0]
        s_y = np.array(foot_step)[:, 1]
        T_c = np.sqrt(self.z_c/self.g)
        C = np.cosh(T_sup/T_c)
        S = np.sinh(T_sup/T_c)
        D = a*(C-1)**2+b*(S/T_c)**2

        x = 0.0
        y = -0.1+0.01  # mumumu
        vx = 0.0
        vy = 0.0
        x_bar = x
        y_bar = y
        vx_bar = vx
        vy_bar = vy
        x_d = 0
        y_d = 0
        vx_d = vx
        vy_d = vy
        T = 0
        n = 0

        # main loop
        while(n+2 < len(s_x)):
            # step 3
            before_T = T
            before_x = x
            before_vx = vx
            before_y = y
            before_vy = vy

            while T < before_T+T_sup:
                tmp_x = x + h*vx
                tmp_vx = vx + h*self.g/self.z_c*(x-self.p_x_star[-1])
                tmp_y = y + h*vy
                tmp_vy = vy + h*self.g/self.z_c*(y-self.p_y_star[-1])
                T = T + h

                x = tmp_x
                vx = tmp_vx
                y = tmp_y
                vy = tmp_vy

                self.x_trj.append(x)
                self.vx_trj.append(vx)
                self.y_trj.append(y)
                self.vy_trj.append(vy)
                self.t_trj.append(T)

            x = C*before_x+T_c*S*before_vx + (1.0-C)*self.p_x_star[-1]
            vx = S/T_c*before_x+C*before_vx + (-S/T_c)*self.p_x_star[-1]
            y = C*before_y+T_c*S*before_vy + (1.0-C)*self.p_y_star[-1]
            vy = S/T_c*before_y+C*before_vy + (-S/T_c)*self.p_y_star[-1]
            print("x", x, y)
            print("vx", vx, vy)

            # step 4
            T = before_T + T_sup
            n = n + 1

            # step 5
            self.p_x.append(self.p_x[n-1]+s_x[n])
            self.p_y.append(self.p_y[n-1]-(-1.0)**n*s_y[n])
            print("p_x", self.p_x[-1], self.p_y[-1])

            # step 6
            x_bar = s_x[n+1]/2
            y_bar = (-1)**n*s_y[n+1]/2
            vx_bar = (C+1)/(T_c*S)*x_bar
            vy_bar = (C-1)/(T_c*S)*y_bar
            print("x_bar", x_bar, y_bar)
            print("vx_bar", vx_bar, vy_bar)

            # step 7
            x_d = self.p_x[-1]+x_bar
            vx_d = vx_bar
            y_d = self.p_y[-1]+y_bar
            vy_d = vy_bar
            print("x_d", x_d, y_d)
            print("vx_d", vx_d, vy_d)

            # step 8
            self.p_x_star.append(-a*(C-1)/D*(x_d-C*x-T_c*S*vx)-b*S/(T_c*D)*(vx_d-S/T_c*x-C*vx))
            self.p_y_star.append(-a*(C-1)/D*(y_d-C*y-T_c*S*vy)-b*S/(T_c*D)*(vy_d-S/T_c*y-C*vy))
            print("p_x_star", self.p_x_star[-1], self.p_y_star[-1])

            # step 9
            print(n, len(s_x))

    def draw(self):
        self.ax = plt.axes()
        for i in range(len(self.p_x)):
            self.ax.add_patch(patches.Rectangle(xy=(self.p_x[i]-self.height/2, self.p_y[i]-self.width/2),
                                                width=self.height, height=self.width, ec='#CCCCCC', fill=False))
            self.ax.add_patch(patches.Rectangle(
                xy=(self.p_x_star[i]-self.height/2, self.p_y_star[i]-self.width/2), width=self.height, height=self.width, ec='#000000', fill=False))
        self.ax.set_aspect('equal')
        plt.plot(self.x_trj, self.y_trj)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.5, 0.5)
        plt.show()


def test1():
    lipModelSimulation = LIPModelSimulation(0.2, 0.1, 0.8)
    lipModelSimulation.run([(0.0, 0.0), (0.0, 0.2), (0.3, 0.2), (0.3, 0.2),
                            (0.3, 0.2), (0.0, 0.2), (0.0, 0.0)], 0.8, 0.005)
    lipModelSimulation.draw()


def main():
    test1()


if __name__ == '__main__':
    main()
