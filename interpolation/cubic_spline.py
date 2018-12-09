import numpy as np
import matplotlib.pyplot as plt


class CubicSplineInterpolation:
    def __init__(self):
        print("# initializing...")

        # for drawing
        self.x_seq = []
        self.y_seq = []

        self.x_cmd_seq = []
        self.y_cmd_seq = []

        print("# initialized.")

    def u_j(self, points, j, x):
        ans = 1.0
        for i, p in enumerate(points):
            if i != j:
                ans *= (x-p[0])/(points[j][0]-p[0])
        return ans

    def run(self, points, x_interval):
        N = len(points)-1
        px = np.array(points)[:, 0]
        py = np.array(points)[:, 1]
        self.x_cmd_seq = px[:]
        self.y_cmd_seq = py[:]

        h = []
        for x0, x1 in zip(px[:-1], px[1:]):
            h.append(x1-x0)
        h = np.array(h)
        assert(len(h) == N)
        v = []  # 1-indexed
        for y0, y1, y2, h0, h1 in zip(py[0:-2], py[1:-1], py[2:], h[:-1], h[1:]):
            v.append(6.*((y2-y1)/h1-(y1-y0)/h0))
        v = np.array(v)
        assert(len(v) == N-1)

        mat_for_u = np.zeros((N-1, N-1))
        for i in range(N-1):
            for j in range(N-1):
                if i == j:
                    mat_for_u[i][j] = 2.*(h[i]+h[i+1])
                elif i == j+1:
                    mat_for_u[i][j] = h[i]
                elif j == i+1:
                    mat_for_u[i][j] = h[j]
        v_for_u = v.reshape(len(v), 1)

        u = np.linalg.inv(mat_for_u).dot(v_for_u)
        u = u.flatten()
        u = np.concatenate([[0.], u, [0.]])

        b = u[:-1]/2.0

        a = (u[1:]-2.0*b)/(6.0*h)

        d = py[:-1]

        c = 1.0/h*(py[1:]-a*h**3-b*h**2-d)
        print(a)
        print(b)
        print(c)
        print(d)

        for i, (x0, y0, x1, y1) in enumerate(zip(px[:-1], py[:-1], px[1:], py[1:])):
            print(x0, x1)
            for x in np.arange(x0, x1, x_interval):
                y = a[i]*(x-x0)**3 + b[i]*(x-x0)**2 + c[i]*(x-x0) + d[i]
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
    cubicSplineInterpolation = CubicSplineInterpolation()
    cubicSplineInterpolation.run([(0., 0.), (0.5, 1.), (0.8, 0.2), (1., 0.)], 0.01)
    cubicSplineInterpolation.draw()


def test2():
    cubicSplineInterpolation = CubicSplineInterpolation()
    cubicSplineInterpolation.run([(0., 0.), (0.1, 0.1), (0.2, 0.1), (0.3, 0.2),
                                  (0.5, 1.), (0.7, 0.2), (0.8, 0.1), (0.9, 0.1), (1.0, 0.0)], 0.01)
    cubicSplineInterpolation.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
