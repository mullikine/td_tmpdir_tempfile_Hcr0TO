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

    data = (6 * x + 2) + np.random.rand(1, 200) * 50
    print(data.shape)
    plt.scatter(range(200), data)
    plt.plot(range(200), output, color="red")

    error = data - output
    meanerror = np.mean(error)
    # print(error.shape)

    # The error sucks
    print(meanerror)

    # d_k = - (2 * error) * x
    # d_b = 2 * error

    lr = 0.000001 # (learning rate)

    epocs = 10000

    for i in range(epocs):
        # 4 batches
        for j in range(4):
            x = np.array(range(50))

            # forward prop
            output = k * x + b


            # after computing output, we compute its error
            e = data[0][j*50:(j+1)*50] - output
            error = (e) ** 2
            meanerror = np.mean(error)
            print("me:", meanerror)

            # if i % 10 == 0:
            #     plt.scatter(range(200), data)
            #     plt.plot(range(200), output, color="red")
            #     plt.show()

            k = k - lr * np.mean(- 2 * (e) * x)
            b = b - lr * (-2 * np.mean(e))

            # k = k - np.mean(- lr * x)
            # b = b - lr * -1

            print("k:", k)
            print("b:", b)


    # plt.show()


    #  np.random.rand(2, 200)

if __name__ == '__main__':
    main()