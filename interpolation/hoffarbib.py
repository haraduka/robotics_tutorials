import numpy as np
import matplotlib.pyplot as plt


class HoffArbibInterpolation:
    def __init__(self, a_cur, v_cur, x_cur):
        print("# initializing...")

        self.t_cur = 0
        self.a_cur = a_cur
        self.v_cur = v_cur
        self.x_cur = x_cur

        # for drawing
        self.t_seq = []
        self.a_seq = []
        self.v_seq = []
        self.x_seq = []
        self.x_cmd_seq = []

        print("# initialized.")

    def run(self, x_cmd, t_interpolation, t_interval):
        assert(len(x_cmd) == int(t_interpolation/t_interval))
        i = 0
        for t_cur in np.arange(0, t_interpolation, t_interval):
            t_remain = t_interpolation - t_cur
            d_a_cmd = - 9.*self.a_cur/t_remain - 36.*self.v_cur/t_remain**2 + 60.*(x_cmd[i]-self.x_cur)/t_remain**3

            self.t_cur += t_interval
            self.a_cur += d_a_cmd*t_interval
            self.v_cur += self.a_cur*t_interval
            self.x_cur += self.v_cur*t_interval

            self.t_seq.append(self.t_cur)
            self.a_seq.append(self.a_cur)
            self.v_seq.append(self.v_cur)
            self.x_seq.append(self.x_cur)
            self.x_cmd_seq.append(x_cmd[i])

            i += 1

    def draw(self):
        self.fig = plt.figure()
        self.a_plot = self.fig.add_subplot(3, 1, 1)
        self.v_plot = self.fig.add_subplot(3, 1, 2)
        self.x_plot = self.fig.add_subplot(3, 1, 3)
        self.a_plot.set_ylabel("a")
        self.v_plot.set_ylabel("v")
        self.x_plot.set_ylabel("x, x_cmd")
        self.a_plot.plot(self.t_seq, self.a_seq, color='b')
        self.v_plot.plot(self.t_seq, self.v_seq, color='b')
        self.x_plot.plot(self.t_seq, self.x_seq, color='b')
        self.x_plot.plot(self.t_seq, self.x_cmd_seq, color='r')
        plt.show()


def test1():
    hoffArbib = HoffArbibInterpolation(0.0, 0.0, 0.0)
    hoffArbib.run(np.repeat(1.0, 100), 1.0, 0.01)
    hoffArbib.draw()

def test2():
    hoffArbib = HoffArbibInterpolation(0.0, 0.0, 0.0)
    hoffArbib.run(np.sin(np.arange(0.0, 1.0, 0.01)), 1.0, 0.01)
    hoffArbib.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
