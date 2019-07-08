import numpy as np
import matplotlib.pyplot as plt

# maximum (average between class)/(variance in class)
# maximum w^TS_Bw/w^TS_Ww
# lagrange multiplier -> w = S_W^-1S_Bw = S_w^-1(m_1-m_2) (m1, m2 = ave in class)

class Fisher:
    def __init__(self):
        print("# initializing...")

        # raw data
        self.x_data1 = []
        self.y_data1 = []

        self.x_data2 = []
        self.y_data2 = []

        # discrimination line
        self.x_v= []
        self.y_v= []

        print("# initialized.")

    def run(self, points1, points2):
        ave1 = np.average(points1, axis=0)
        ave2 = np.average(points2, axis=0)
        self.min_p1 = np.min(points1, axis=0)
        self.max_p1 = np.max(points1, axis=0)
        self.min_p2 = np.min(points2, axis=0)
        self.max_p2 = np.max(points2, axis=0)

        S_W = (points1-ave1).T.dot(points1-ave1) + (points2-ave2).T.dot(points2-ave2)
        w = (ave1-ave2).dot(np.linalg.inv(S_W).T)

        self.x_data1 = points1[:, 0]
        self.y_data1 = points1[:, 1]
        self.x_data2 = points2[:, 0]
        self.y_data2 = points2[:, 1]

        self.x_v1 = (np.arange(200)-100)+(ave1[0]+ave2[0])/2
        self.y_v1 = -w[0]/w[1]*self.x_v1+(ave1[1]+ave2[1])/2

    def draw(self):
        self.fig = plt.figure()
        plt.axis('equal')
        plt.xlim(min(self.min_p1[0], self.min_p2[0]), max(self.max_p1[0], self.max_p2[0]))
        plt.ylim(min(self.min_p1[1], self.min_p2[1]), max(self.max_p1[1], self.max_p2[1]))
        self.xy_plot = self.fig.add_subplot(1, 1, 1)
        self.xy_plot.set_xlabel("x")
        self.xy_plot.set_ylabel("y")
        self.xy_plot.scatter(self.x_data1, self.y_data1, color='r')
        self.xy_plot.scatter(self.x_data2, self.y_data2, color='b')
        self.xy_plot.plot(self.x_v1, self.y_v1, color='g')
        plt.show()

def test1():
    fisher = Fisher()
    fisher.run(np.random.randn(20, 2).dot(np.array([[3, 1], [1, 3]]))+[5, 5], np.random.randn(20, 2).dot(np.array([[3, 1], [1, 3]]))+[-5, -5])
    fisher.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
