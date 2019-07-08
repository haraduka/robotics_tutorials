import numpy as np
import matplotlib.pyplot as plt

# E = a^TVar(X)a - lambda*(a^Ta-1)
# -> lagrange multiplier -> Var(X)a = lambda*a -> eigenvalue decomposition

class PCA:
    def __init__(self):
        print("# initializing...")

        # raw data
        self.x_data = []
        self.y_data = []

        # first axis
        self.x_v1= []
        self.y_v1= []

        # second axis
        self.x_v2= []
        self.y_v2= []

        print("# initialized.")

    def run(self, points):
        ave = np.average(points, axis=0)
        min_p = np.min(points, axis=0)
        max_p = np.max(points, axis=0)
        V = (points-ave).T.dot(points-ave)
        value, v = np.linalg.eig(V)
        print(value)
        print(v)

        self.x_data = points[:, 0]
        self.y_data = points[:, 1]

        self.x_v1 = (value[0]/(value[0]+value[1]))*(max_p[0]-min_p[0])*0.01*(np.arange(100)-50)+ave[0]
        self.y_v1 = v[1][0]/v[0][0]*(self.x_v1-ave[0])+ave[1]

        self.x_v2 = (value[1]/(value[0]+value[1]))*(max_p[0]-min_p[0])*0.01*(np.arange(100)-50)+ave[0]
        self.y_v2 = v[1][1]/v[0][1]*(self.x_v2-ave[0])+ave[1]

    def draw(self):
        self.fig = plt.figure()
        plt.axis('equal')
        self.xy_plot = self.fig.add_subplot(1, 1, 1)
        self.xy_plot.set_xlabel("x")
        self.xy_plot.set_ylabel("y")
        self.xy_plot.scatter(self.x_data, self.y_data, color='r')
        self.xy_plot.plot(self.x_v1, self.y_v1, color='b')
        self.xy_plot.plot(self.x_v2, self.y_v2, color='g')
        plt.show()

def test1():
    pca = PCA()
    pca.run(np.random.randn(20, 2).dot(np.array([[3, 1], [1, 2]])))
    pca.draw()

def test2():
    pca = PCA()
    pca.run(np.random.randn(20, 2).dot(np.array([[0, 3], [5, 2]])))
    pca.draw()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
