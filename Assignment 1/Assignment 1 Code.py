import random
import numpy as np
import matplotlib.pyplot as plt

def getPrediction(thetha0, thetha1, input):
    return (thetha0 + (thetha1 * input))

def getCost(thetha0, thetha1, data):
    sum0 = 0
    n = (float)(len(data))
    for i in range((int)(n)):
        x = data[i][0]
        y = data[i][1]
        prediction = getPrediction(thetha0, thetha1, x)
        sum0 += (prediction - y) ** 2
    
    return sum0 / n

def getGradientDescentStepValue(Thetha0, Thetha1, data):
    sum0 = 0
    sum1 = 0
    n = (float)(len(data))
    for i in range((int)(n)):
        x = data[i][0]
        y = data[i][1]
        prediction = getPrediction(Thetha0, Thetha1, x)
        sum0 += (prediction - y)
        sum1 += (prediction - y) * x
    gradient0 = (sum0 / n)
    gradient1 = (sum1 / n)
    return [gradient0, gradient1]

if __name__ == '__main__':
    #area = [5,6,7,9,10,12,20] # Area in Marla
    #price = [2041666,3200000,3775000,4200000,5016667,4950000,8100000] # Price in Pakistani Ruppees

    #data = [[1360, 2041666], [1632, 3200000], [1904, 3775000], [2448, 4200000], [2720, 5016667], [3264, 5516667], [5440, 8100000]]
    #data = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70]]
    #data = [[127, 305000], [106, 275000], [106, 244000], [102, 199900], [171, 319000], [139, 265000], [129, 309500], [125, 289000]]

    with open('data.txt') as f:
        w, h = [float(x) for x in next(f).split()]
        data = [[float(x) for x in line.split()] for line in f]

    delta = []
    thethas = []

    thetha0 = random.random()
    thetha1 = random.random()
    alpha = 0.009
    delta0 = 0
    delta1 = 0
    
    thethas.append([thetha0, thetha1])
    
    iteration = 10000
    
    for i in range(iteration):
        print('iteration: ', i)
        print('thetha0: ', thetha0)
        print('thetha1: ', thetha1)
        print('Error: ', getCost(thetha0,thetha1,data))
        delta0, delta1 = getGradientDescentStepValue(thetha0, thetha1, data)
        thetha0 -= (alpha * delta0)
        thetha1 -= (alpha * delta1)
        delta.append([delta0, delta1])
        thethas.append([thetha0, thetha1])
        print('Gradient Descent 0 Value: ', delta0)
        print('Gradient Descent 1 Value: ', delta1)
        print()
    
    arr = np.asarray(delta)
    plt.plot(arr)
    plt.ylabel('Gradient Descent Value')
    plt.xlabel('Iterations')
    plt.legend(('thetha0', 'thetha1'))

    plt.show()
    
    arr = np.asarray(thethas)
    plt.plot(arr)
    plt.ylabel('Thetha Value')
    plt.xlabel('Iterations')
    plt.legend(('thetha0', 'thetha1'))
    
    plt.show()
    
    print('Using the updated parameters, the value of 4: ', getPrediction(thetha0, thetha1, 4))
