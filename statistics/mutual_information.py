import numpy as np
import matplotlib.pyplot as plt

# squared-loss MI
# p(x)p(y)(p(x, y)/p(x)p(y)-1)^2

class MutualInformation:
    def __init__(self):
        print("# initializing...")
        print("# initialized.")

    def run(self, data1, data2):
        N = len(data1)
        B =  10
        D = data1.shape[1]*data2.shape[1]
        S = 1.0
        G = np.zeros((B, B), dtype=np.float64)
        index = np.random.permutation(np.arange(N))
        for i in xrange(N):
            for j in xrange(N):
                x1 = data1[i]
                x2 = data2[j]
                g = np.array([], dtype=np.float64)
                for b in xrange(B):
                    g = np.append(g, np.exp(-np.linalg.norm(x1-data1[index[b]])**2/(2*S**2))*np.exp(-np.linalg.norm(x2-data2[index[b]])**2/(2*S**2)))
                G += g[np.newaxis].T.dot(g[np.newaxis])/N**2
        h = np.zeros((B, 1), dtype=np.float64)
        for i in xrange(N):
            x1 = data1[i]
            x2 = data2[i]
            g = np.array([], dtype=np.float64)
            for b in xrange(B):
                g = np.append(g, np.exp(-np.linalg.norm(x1-data1[index[b]])**2/(2*S**2))*np.exp(-np.linalg.norm(x2-data2[index[b]])**2/(2*S**2)))
            h += g[np.newaxis].T/N
        lam = 1.0e-9
        LSMI = np.linalg.inv(G+lam*np.identity(B)).dot(h)
        SMI = (h.T.dot(LSMI)-1)/2.
        # print(LSMI)
        print(SMI)

def test1():
    mi = MutualInformation()
    x = np.random.randn(100, 2)
    y = np.random.randn(100, 3)
    mi.run(x, y)
    mi.run(x, x*2+10)
    mi.run(x, x**3)
    mi.run(x, np.cos(x))

def main():
    test1()

if __name__ == '__main__':
    main()
