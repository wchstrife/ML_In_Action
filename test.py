from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    m = 100
    dataIndex = range(m)
    print(dataIndex)
    for i in range(m):
            randIndex = int(random.uniform(0, len(dataIndex)))
            print(dataIndex[randIndex])
            