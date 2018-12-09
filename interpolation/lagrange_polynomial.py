import numpy as np
import matplotlib.pyplot as plt


class LagrangePolynomialInterpolation:
    def __init__(self):
        print("# initializing...")

        # for drawing
        self.x_seq = []
        self.y_seq = []
        self.x_cmd_seq = []
        self.y_cmd_seq = []

        print("# initialized.")

    def l_j(self, points, j, x):
        ans = 1.0
        for i, p in enumerate(points):
            if i != j:
                ans *= (x-p[0])/(points[j][0]-p[0])
        return ans

    def run(self, points, x_interval):
        px = np.array(points)[:, 0]
        py = np.array(points)[:, 1]
        self.x_cmd_seq = px[:]
        self.y_cmd_seq = py[:]

        for x in np.arange(points[0][0], points[-1][0], x_interval):
            y = 0.
            for j in range(len(points)):
                y += points[j][1]*self.l_j(points, j, x)
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
    lagrangePolynomialInterpolation = LagrangePolynomialInterpolation()
    lagrangePolynomialInterpolation.run([(0., 0.), (0.5, 1.), (0.8, 0.2), (1., 0.)], 0.01)
    lagrangePolynomialInterpolation.draw()


def test2():
    lagrangePolynomialInterpolation = LagrangePolynomialInterpolation()
    lagrangePolynomialInterpolation.run([(0., 0.), (0.1, 0.1), (0.2, 0.1), (0.3, 0.2),
                                         (0.5, 1.), (0.7, 0.2), (0.8, 0.1), (0.9, 0.1), (1.0, 0.0)], 0.01)
    lagrangePolynomialInterpolation.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
