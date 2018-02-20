import numpy as np
import matplotlib.pyplot as plt
from Data import Data

"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


# ===== Helper Function for Plotting - DO NOT EDIT =============
def boundary(w):
    boundary_x1 = np.array([0,1])
    boundary_x2 = -(w[0]*boundary_x1+w[2])/w[1]
    midpt = np.array([0.5,-(w[0]*0.5+w[2])/w[1]])
    vec = (w[:2])/np.linalg.norm(w[:2])
    endpt = midpt + 0.1*vec
    xvec = np.array([midpt[0],endpt[0]])
    yvec = np.array([midpt[1],endpt[1]])
    return (boundary_x1,boundary_x2), (xvec,yvec)

def plot_boundary(w, quiet=True):
    bndry, direction = boundary(w)
    b1,b2 = bndry
    if not quiet:
        color = 'black'
        plt.plot(b1,b2,'-',color=color,label='decision\nboundary')
    else:
        color = 'lightgray'
        plt.plot(b1,b2,'-',color=color)
    xvec,yvec = direction
    plt.plot(xvec,yvec,'o-',color=color)

# =============================================================

class LogisticRegression:
    def __init__(self,  alpha=1e-2):
        self.w = np.array([0,-1,.5]) # You can use these as the initial values for the weights

    def fit(self, X, Y, epochs=10000):
        """
        Make use of self.w to ensure changes in the weight are reflected globally
        :param X: input features, shape:(N,3); Note: Ones for the bias have already been appended for your convenience
        :param Y: target, shape:(N,)
        :return: None

        IMP: Make use of self.w to ensure changes in the weight are reflected globally.
            We will be making use of get_params and set_params to check for correctness
        """
        step = 0.0001 # set the value for step parameter
        for t in range(epochs):
            # WRITE the required CODE HERE to update the parameters using gradient descent
            # make use of self.loss_grad() function
            if t % 100 == 0:
                plot_boundary(self.w)

            self.w = self.w + step * self.loss_grad(self.w, X, Y)

        print(self.loss(self.w, X, Y))
    def predict(self, X):
        """
        Return your predictions
        :param X: inputs, shape:(N,3)
        :return: predictions, shape:(N,)
        """

        # WRITE the required CODE HERE and return the computed values
        return X.dot(self.w)

    def loss(self, w, X, Y):
        """
        :param W: weights, shape:(3,)
        :param X: input, shape:(N,3)
        :param Y: target, shape:(N,)
        :return: scalar loss value
        """
        sig = 1/(1+np.exp(-(X.dot(self.w))))
        losssum = 0

        for i in range(Y.shape[0]):

            losssum += -Y[i]*(np.log(sig))- (1 - Y[i]) *(np.log(1 - sig))
        # WRITE the required CODE HERE and return the computed values

        return np.nansum(losssum) / Y.shape[0]

    def loss_grad(self, w, X, y):
        """
        Compute the gradient of the loss.
        (Function will be tested only for gradient descent)
        :param W: weights, shape:(3,)
        :param X: input, shape:(N,3)
        :param Y: target, shape:(N,)
        :return: vector of size (3,) containing gradients for each weight
        """
        # WRITE the required CODE HERE and return the computed values

        sig = 1/(1+np.exp(-(X.dot(self.w))))
        grad = (sig - y).T.dot(X)
        return grad

    def get_params(self):
        """
        ********* DO NOT EDIT ************
        :return: the current parameters value
        """
        return self.w

    def set_params(self, w):
        """
        ********* DO NOT EDIT ************
        Will be used to set the value of weights externally
        :param w:
        :return: None
        """
        self.w = w
        return 0


if __name__ == '__main__':
    # Get data
    data = Data()
    X, y, X_train, y_train, X_test, y_test = data.get_logistic_regression_data()
    N = np.shape(X)[0]

    # Logistic regression with gradient descent
    model = LogisticRegression()
    model.fit(X, y)
    y = model.predict(X)
    w_grad = model.get_params()

    # Plot the results
    plt.plot(X[:N // 2, 0], X[:N // 2, 1], 'r+', label='pos')
    plt.plot(X[N // 2:, 0], X[N // 2:, 1], 'b_', label='neg')
    plot_boundary(w_grad, quiet=False)
    plt.legend()
    plt.axis('square')
    plt.axis([0, 1, 0, 1])
    plt.savefig('figures/Q2.png')
    plt.close()

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.loss(model.w, X_test, y_test))
