import numpy as np
import matplotlib.pyplot as plt


class BsplineInterpolation:
    def __init__(self, n):
        print("# initializing...")

        self.n = n  # degree

        # for drawing
        self.x_seq = []
        self.y_seq = []

        self.x_cmd_seq = []
        self.y_cmd_seq = []

        print("# initialized.")

    def uniform_knot_vector(self):
        ts = []
        for j in range(self.m):
            ts.append(0. + (1.-0.)/(self.m-1.)*j)
        return np.array(ts)

    def open_uniform_knot_vector(self):
        ts = []
        for j in range(self.m):
            if j < self.n+1:
                ts.append(0.)
            elif j >= self.m-(self.n+1):
                ts.append(1.)
            else:
                ts.append(0. + (1.0-0.)/(self.m+1.0-2*(self.n+1.))*(j-self.n))
        return np.array(ts)

    def basic_function(self, ts, t, j, k):
        a = b = 0.
        if k == 0:
            if ts[j] <= t and t < ts[j+1]:
                return 1.
            else:
                return 0.
        else:
            if ts[j+k]-ts[j] != 0:
                a = (t-ts[j])/(ts[j+k]-ts[j])*self.basic_function(ts, t, j, k-1)
            if ts[j+k+1]-ts[j+1] != 0:
                b = (ts[j+k+1]-t)/(ts[j+k+1]-ts[j+1])*self.basic_function(ts, t, j+1, k-1)
            return a+b

    def run(self, points, t_interval):
        self.p = len(points)
        self.m = self.p + self.n + 1
        px = np.array(points)[:, 0]
        py = np.array(points)[:, 1]
        self.x_cmd_seq = px[:]
        self.y_cmd_seq = py[:]

        ts = self.open_uniform_knot_vector()
        assert(len(ts) == self.m)
        print(ts)

        for t in np.arange(0., 1., t_interval):
            x = y = 0.
            for i in range(0, self.m-self.n-2+1, 1):
                b_i_n_t = self.basic_function(ts, t, i, self.n)
                x += points[i][0]*b_i_n_t
                y += points[i][1]*b_i_n_t
            self.x_seq.append(x)
            self.y_seq.append(y)

    def draw(self):
        self.fig = plt.figure()
        self.xy_plot = self.fig.add_subplot(1, 1, 1)
        self.xy_plot.set_xlabel("x")
        self.xy_plot.set_ylabel("y")
        self.xy_plot.plot(self.x_seq, self.y_seq, color='b')
        self.xy_plot.plot(self.x_cmd_seq, self.y_cmd_seq, color='r')
        plt.show()


def test1():
    bsplineInterpolation = BsplineInterpolation(2)
    bsplineInterpolation.run([(0., 0.), (0.5, 1.), (0.8, 0.2), (1., 0.)], 0.01)
    bsplineInterpolation.draw()


def test2():
    bsplineInterpolation = BsplineInterpolation(2)
    bsplineInterpolation.run([(0., 0.), (0.1, 0.1), (0.2, 0.1), (0.3, 0.2),
                              (0.5, 1.), (0.7, 0.2), (0.8, 0.1), (0.9, 0.1), (1.0, 0.0)], 0.01)
    bsplineInterpolation.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
