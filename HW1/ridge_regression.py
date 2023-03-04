import numpy as np
import matplotlib.pyplot as plt
from load_data import load_dataset

def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:

    n, d = x.shape

    regMatrix = _lambda*np.eye(d)
    regMatrix[0, 0] = 0

    theta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    return theta
    raise NotImplementedError("Your Code Goes Here")


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:

    return np.dot(x,w)


    raise NotImplementedError("Your Code Goes Here")

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:

    return np.eye(num_classes)[y]
    raise NotImplementedError("Your Code Goes Here")


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    print(np.shape(x_train))
    plt.imshow(np.reshape(x_train[50],(28,28)))
    plt.show()
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10) # sorunsuz çalışıyor

    for i in range(3):
        print(y_train[i])
        print(y_train_one_hot[i])

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)
    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    for i in range(20):
        print(y_test[i])
        print(y_test_pred[i])


if __name__ == "__main__":
    main()
