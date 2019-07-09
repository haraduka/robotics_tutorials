import numpy as np
import matplotlib.pyplot as plt

# E = a^TVar(X)a - lambda*(a^Ta-1)
# -> lagrange multiplier -> Var(X)a = lambda*a -> eigenvalue decomposition

class KernelPCA:
    def __init__(self):
        print("# initializing...")

        # raw data
        self.x_data1 = []
        self.y_data1 = []

        self.x_data2 = []
        self.y_data2 = []

        self.x_data3 = []
        self.y_data3 = []

        # kernelized raw data
        self.x_kdata1 = []
        self.y_kdata1 = []

        self.x_kdata2 = []
        self.y_kdata2 = []

        self.x_kdata3 = []
        self.y_kdata3 = []


        print("# initialized.")

    def run(self, points1, points2, points3, kernel_func):
        points = np.concatenate([points1,points2,points3], axis=0)
        N = len(points)
        K = np.zeros((N, N), dtype=np.float32)
        for i in xrange(N):
            for j in xrange(N):
                K[i, j] = kernel_func(points[i], points[j])
        ones = np.ones((N, N), dtype=np.float32)/N
        gram = K-ones.dot(K)-K.dot(ones)+ones.dot(K).dot(ones)
        value, v = np.linalg.eig(gram)
        index = np.argsort(value)
        x1 = index[-1]
        x2 = index[-2]
        self.x_data1 = points1[:, 0]
        self.y_data1 = points1[:, 1]
        self.x_data2 = points2[:, 0]
        self.y_data2 = points2[:, 1]
        self.x_data3 = points3[:, 0]
        self.y_data3 = points3[:, 1]
        for i in xrange(len(points1)):
            self.x_kdata1.append(v[i, x1])
            self.y_kdata1.append(v[i, x2])
        for i in xrange(len(points2)):
            self.x_kdata2.append(v[len(points1)+i, x1])
            self.y_kdata2.append(v[len(points1)+i, x2])
        for i in xrange(len(points3)):
            self.x_kdata3.append(v[len(points1)+len(points2)+i, x1])
            self.y_kdata3.append(v[len(points1)+len(points2)+i, x2])


    def draw(self):
        self.fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.x_data1, self.y_data1, color='r')
        plt.scatter(self.x_data2, self.y_data2, color='g')
        plt.scatter(self.x_data3, self.y_data3, color='b')
        plt.subplot(2, 1, 2)
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(self.x_kdata1, self.y_kdata1, color='r')
        plt.scatter(self.x_kdata2, self.y_kdata2, color='g')
        plt.scatter(self.x_kdata3, self.y_kdata3, color='b')
        plt.show()

def test1():
    kernelPCA = KernelPCA()
    kernelPCA.run(np.random.randn(20, 2).dot(np.array([[2, 0], [0, 1]]))+[5.0, 0.0],
            np.random.randn(20, 2).dot(np.array([[1, 0], [0, 2]]))+[5.0, 5.0],
            np.random.randn(20, 2).dot(np.array([[2, 0], [0, 1]]))+[0.0, 5.0],
            #lambda x, y: (x.dot(y)+1.0)**2)
            lambda x, y: np.exp(-np.linalg.norm(x-y)**2/1.0))
    kernelPCA.draw()

def test2():
    kernelPCA = KernelPCA()
    kernelPCA.run(
            np.array([[(0+x[0])*np.cos(100*x[1]), (0+x[0])*np.sin(100*x[1])] for x in np.random.randn(50, 2)]),
            np.array([[(10+x[0])*np.cos(100*x[1]), (10+x[0])*np.sin(100*x[1])] for x in np.random.randn(50, 2)]),
            np.array([[(20+x[0])*np.cos(100*x[1]), (20+x[0])*np.sin(100*x[1])] for x in np.random.randn(50, 2)]),
            lambda x, y: (x.dot(y)+1.0)**2)
    kernelPCA.draw()

def main():
    # test1()
    test2()


if __name__ == '__main__':
    main()
