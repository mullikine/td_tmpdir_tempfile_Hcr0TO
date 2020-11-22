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
    # forward prop
    output = k * x + b

    data = np.linspace(0, 200, num=200) + np.random.rand(1, 200) * 50
    print(data.shape)
    plt.scatter(range(200), data)
    plt.plot(range(200), output, color="red")

    error = data - output
    meanerror = np.mean(error)
    # print(error.shape)

    # The error sucks
    print(meanerror)

    # d_k = -x (error)
    # d_b = -1 (error)

    lr = 0.0001 # (learning rate)

    epocs = 100

    for i in range(epocs):
        # forward prop
        output = k * x + b

        # after computing output, we compute its error
        error = (data - output) ** 2
        meanerror = np.mean(error)
        print("me:", meanerror)

        k = k - np.mean(lr * -x)
        b = b - lr * -1
        print("k:", k)
        print("b:", b)


    # plt.show()


    #  np.random.rand(2, 200)

if __name__ == '__main__':
    main()