import numpy as np

class Data:
    """
    ***********DO NOT EDIT THIS CLASS******************
    """
    def __init__(self):
        pass

    def get_linear_regression_data(self, N=10):
        np.random.seed(1234)
        X = np.hstack((10 * np.random.rand(N, 1), np.ones((N, 1))))
        X = X[np.argsort(X[:, 0])]
        w = np.array([1, 2])
        y = np.dot(X, w)
        y = y ** 2. + 2 * np.random.rand(N)
        return (X, y)

    def get_logistic_regression_data(self, N=100):
        Xp = np.hstack((0.5 * np.random.rand(N // 2, 2) + np.array([0.1, 0.4]), np.ones((N // 2, 1))))
        Xm = np.hstack((0.5 * np.random.rand(N // 2, 2) + np.array([0.4, 0.1]), np.ones((N // 2, 1))))
        X = np.vstack((Xp, Xm))
        y = np.vstack((np.ones((N // 2, 1)), np.zeros((N // 2, 1)))).flatten()
        inds = range(N)
        X = X[inds]
        y = y[inds]
        split = int(0.6*N)
        X_train = X[:split]
        y_train = y[:split]
        X_test = X[split:]
        y_test = y[split:]
        return (X, y, X_train, y_train, X_test, y_test)

    def get_decision_tree_data(self):
        attribute_names = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
        attribute_values = [[True, False],
                            [True, False],
                            [True, False],
                            [True, False],
                            ['None', 'Some', 'Full'],
                            ['$', '$$$'],
                            [True, False],
                            [True, False],
                            ['French', 'Thai', 'Burger', 'Italian'],
                            ['0-10', '10-30', '30-60', '>60']]
        examples = [[True, False, False, True, 'Some', '$$$', False, True, 'French', '0-10', True],
                    [True, False, False, True, 'Full', '$', False, False, 'Thai', '30-60', False],
                    [False, True, False, False, 'Some', '$', False, False, 'Burger', '0-10', True],
                    [True, False, True, True, 'Full', '$', False, False, 'Thai', '10-30', True],
                    [True, False, True, False, 'Full', '$$$', False, True, 'French', '>60', False],
                    [False, True, False, True, 'Some', '$$', True, True, 'Italian', '0-10', True],
                    [False, True, False, False, 'None', '$', True, False, 'Burger', '0-10', False],
                    [False, False, False, True, 'Some', '$$', True, True, 'Thai', '0-10', True],
                    [False, True, True, False, 'Full', '$', True, False, 'Burger', '>60', False],
                    [True, True, True, True, 'Full', '$$$', False, True, 'Italian', '10-30', False],
                    [False, False, False, False, 'None', '$', False, False, 'Thai', '0-10', False],
                    [True, True, True, True, 'Full', '$', False, False, 'Burger', '30-60', True]]

        return examples, attribute_names, attribute_values

    def get_kmeans_data(self):
        X1 = np.hstack((np.random.normal(0.75, 0.075, 50)[:, None], np.random.normal(0.75, 0.025, 50)[:, None]))
        X2 = np.hstack((np.random.normal(0.65, 0.125, 50)[:, None], np.random.normal(0.35, 0.1, 50)[:, None]))
        X3 = np.hstack((np.random.normal(0.25, 0.05, 50)[:, None], np.random.normal(0.65, 0.075, 50)[:, None]))
        X = np.vstack((X1, X2, X3))

        return X

    def get_bayesian_data(self):
        limes = np.array(range(11))
        cherries = 0

        return limes, cherries