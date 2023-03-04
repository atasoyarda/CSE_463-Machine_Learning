# Note for this import to work you need to call python from root directory.
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from homeworks.ridge_regression_mnist import ridge_regression
from utils import load_dataset, problem

RNG = np.random.RandomState(seed=546)

Dataset = Tuple[np.ndarray, np.ndarray]


@problem.tag("hw1-B")
def transform_data(x: np.ndarray, G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Transform data according to the problem

    Args:
        x (np.ndarray): An array of shape (n, d). Observations.
        G (np.ndarray): Matrix G of shape (d, p), as specified in problem.
        b (np.ndarray): Array b of shape (p,), as specified in problem.

    Returns:
        np.ndarray: Cosine tranformation of input data x, after multiplication with G and addition of b.

    Note:
        - Do not call RNG in this function. G and b should be initialized in the main function.
            You will need to save them for part b.
    """
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-B")
def split_into_validation(
    x: np.ndarray, y: np.ndarray, fraction_train: float = 0.8
) -> Tuple[Dataset, Dataset]:
    """Splits training dataset into training and validation subsets.

    Args:
        x (np.ndarray): An array of shape (n, d). Observations of training set.
        y (np.ndarray): An array of shape (n,). Targets of training set.
        fraction_train (float, optional): Fraction of original (input) training, that should be kept in resulting training set.
            Remainder should go to validation. Defaults to 0.8.

    Returns:
        Tuple[Dataset, Dataset]: Tuple of two datasets. Each dataset is itself a tuple of 2 numpy array (observation and targets).
            Thus return should look similar to this: `return (x_train, y_train), (x_val, y_val)`
    """
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-B", start_line=6)
def main():
    """Main function of the problem.
    It loads in data. You should perform a hyperparameter search over p, save the best performing weight, G and b.
    Then plots training and validation error as a function of p.
    Then, for the best p, report training, validation errors, as well test error with confidence interval around it.
    """
    # Load dataset and split train into train & validation
    (x, y), (x_test, y_test) = load_dataset("mnist")
    (x_train, y_train), (x_val, y_val) = split_into_validation(x, y, fraction_train=0.8)
    # Convert targets to one hot encoding
    y_train_one_hot = ridge_regression.one_hot(y_train, 10)
    ps = [10, 20, 40, 80, 160, 320, 640, 1000, 2000, 4000]  # Use these ps for search
    raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
