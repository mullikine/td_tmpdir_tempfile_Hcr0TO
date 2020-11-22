import numpy as np
import matplotlib.pyplot as plt

# def neuron(x):
#     k = 0
#     b = 0
#     # return sigmoid(x * k + b)
#     return x * k + b

def linear_classifier(k, b, x):
    # return sigmoid(x * k + b)
    return x * k + b

# def sigmoid(x):
#     return 1.0 / (exp(-x) + 1.0)

def main():
    k = 2
    b = 0

    x = np.array(range(200))
    y = k * x + b

    data = np.linspace(0, 200, num=200) + np.random.rand(1, 200) * 50
    print(data.shape)
    plt.scatter(range(200), data)
    plt.plot(range(200), y, color="red")

    error = data - y
    meanerror = np.mean(error)
    # print(error.shape)

    # The error sucks
    print(meanerror)

    # d_k = -x (error)
    # d_b = -1 (error)

    lr = 0.001 # (learning rate)

    
    k = k - lr * -
    b = b - lr * -1


    # plt.show()


    #  np.random.rand(2, 200)

if __name__ == '__main__':
    main()