from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from load_data import load_dataset

RNG = np.random.RandomState(seed=446)
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def load_2_7_mnist() -> Dataset:
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    train_idxs = np.logical_or(y_train == 2, y_train == 7)
    test_idxs = np.logical_or(y_test == 2, y_test == 7)

    y_train_2_7 = y_train[train_idxs]
    y_train_2_7 = np.where(y_train_2_7 == 7, 1, -1)

    y_test_2_7 = y_test[test_idxs]
    y_test_2_7 = np.where(y_test_2_7 == 7, 1, -1)

    """
    xtrain 60000-784
    xtest 10000-784
    ytrain 60000,
    ytest 10000,
    y_train_2_7 12223,
    y_test_2_7 2060,

    """

    return (x_train[train_idxs], y_train_2_7), (x_test[test_idxs], y_test_2_7)


class LogisticRegression:

    def __init__(self, reg_lambda=1e-3):

        self.reg_lambda = reg_lambda
        self.weight = None
        self.bias = 0
        self.delta = 1e-4
        
     
    def mu(self, X, Y):
        return 1 / (1 + np.exp(-Y * (self.bias + X.dot(self.weight))))

    def loss(self, X, Y):
        return np.mean(np.log(1 + np.exp(-Y * (self.bias + X.dot(self.weight))))) + self.reg_lambda * self.weight.dot(
            self.weight)  

    def gradient_J_weight(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.mean(((self.mu(X, Y) - 1) * Y)[:, None] * X, axis=0) + (2 * self.reg_lambda * self.weight)
    
    def gradient_J_bias(self, X: np.ndarray, Y: np.ndarray) -> float:
        return np.mean(((self.mu(X, Y) - 1) * Y), axis=0)

    def predict(self,X):
        return np.sign(self.bias + X.dot(self.weight))
                
    def misclassification_error(self, X: np.ndarray, Y: np.ndarray) -> float:
        
        Y_Pred = self.predict(X)
        return 1 - np.mean(Y == Y_Pred)

    def step(self, X, Y, learning_rate: float = 1e-2):
            self.weight -= learning_rate * self.gradient_J_weight(X, Y)
            self.bias -= learning_rate * self.gradient_J_bias(X, Y)
    
    def train(self, X, Y, X_test, Y_test, learning_rate: 1e-2, epochs: 30, batchsize=100):

        if self.weight is None:
            self.weight = np.zeros(X.shape[1])

        history={
                "weight_history":[],
                "bias_history":[],
                "J_history":[],
                "J_test_history":[],
                "training_errors":[],
                "testing_errors":[]
                    }   
        i = 0            
        while i == 0 \
                or np.linalg.norm(self.weight - history["weight_history"][-1]) > self.delta \
                and i < epochs: 

            batch = np.random.choice(X.shape[0], batchsize)
            Xx = X[batch]
            Yy = Y[batch]

            i += 1
            history["weight_history"].append(np.copy(self.weight))
            history["bias_history"].append(self.bias)

            self.step(Xx,Yy)

            history["J_history"].append(self.loss(X, Y))
            history["J_test_history"].append(self.loss(X_test, Y_test))
            history["training_errors"].append(np.copy(self.misclassification_error(X, Y)))
            history["testing_errors"].append(np.copy(self.misclassification_error(X_test, Y_test)))  
            
        return history


def main():

    

    (X_train, Y_train), (X_test, Y_test) = load_2_7_mnist()

    
    model = LogisticRegression()
    history = model.train(X_train, Y_train, X_test, Y_test,1e-2, 1000, X_train.shape[0])

    plt.plot(history["J_history"],  label='Training Loss')
    #plt.plot(history["J_test_history"], label='Testing Loss')
    plt.title("Q3B1")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(history["training_errors"], label='Training Error')
    plt.plot(history["testing_errors"], label='Testing Error')
    plt.title("Q3B2")
    plt.xlabel("Epochs")
    plt.ylabel('Error')
    plt.show()

    model = LogisticRegression()
    history2 = model.train(X_train, Y_train, X_test, Y_test, 1e-2, 1000, 1)


    plt.plot(history2["J_history"],  label='Training Loss')
    plt.plot(history2["J_test_history"], label='Testing Loss')
    plt.title("Q3C1")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    plt.plot(history2["training_errors"], label='Training Error')
    plt.plot(history2["testing_errors"], label='Testing Error')
    plt.title("Q3C2")
    plt.xlabel("Epochs")
    plt.ylabel('Error')
    plt.show()

    model = LogisticRegression()
    history3 =model.train(X_train, Y_train, X_test, Y_test, 1e-2, 1000, 100)


    plt.plot(history3["J_history"], label='Training Loss')
    plt.plot(history3["J_test_history"], label='Testing Loss')
    plt.title("Q3D1")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    plt.plot(history3["training_errors"], label='Training Error')
    plt.plot(history3["testing_errors"], label='Test Error')
    plt.title("Q3D2")
    plt.xlabel("Epochs")
    plt.ylabel('Error')
    plt.show()

if __name__ == '__main__':
    main()