
import numpy as np

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):

        self.degree = degree
        self.regLambda = regLambda


    def polyfeatures(self,X,degree) :
        n=len(X)
        shape=(n,degree)
        pf = np.empty(shape)
        for i in range(n):
            for j in range(0,degree):
                pf[i][j] = X[i] ** (j+1)
        return pf

        
    def fit(self, X, y):

        xf = self.polyfeatures(X, self.degree)
        self.mean = np.mean(xf, 0)
        self.std = np.std(xf, 0) + 1
        xf = (xf - self.mean) / self.std
        n = len(X)
        xf = np.c_[np.ones([n, 1]), xf]
        n,d = xf.shape

        regMatrix = self.regLambda * np.eye(d)
        regMatrix[0, 0] = 0
        self.theta = np.linalg.pinv(xf.T.dot(xf) + regMatrix).dot(xf.T).dot(y)
        
    def predict(self, X):

        xf = self.polyfeatures(X, self.degree)
        xf = (xf - self.mean) / self.std
        n = len(X)
        xf = np.c_[np.ones((n,1)),xf]

        return xf.dot(self.theta)

    def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:

        return  np.square(np.subtract(a, b)).mean()
        
    def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
   
        n = len(Xtrain);
        
        errorTrain = np.zeros((n))
        errorTest = np.zeros((n))

        for i in range(0, n):
            model = PolynomialRegression(degree=degree, regLambda=regLambda)
            model.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])

            errorTrain[i] = PolynomialRegression.mean_squared_error(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
            errorTest[i] = PolynomialRegression.mean_squared_error(model.predict(Xtest), Ytest)

        return (errorTrain, errorTest)