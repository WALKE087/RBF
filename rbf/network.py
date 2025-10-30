import numpy as np

class RBF_Network:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.num_centers = 2
        self.centers = None
        self.weights = None
        self.error_optimo = 0.1
        self.training_history = []

    def rbf_activation(self, distance):
        if distance == 0:
            return 0
        return (distance ** 2) * np.log(distance)

    def euclidean_distance(self, x, center):
        return np.sqrt(np.sum((x - center) ** 2))

    def initialize_centers(self, X):
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        self.centers = np.random.uniform(min_vals, max_vals, (self.num_centers, X.shape[1]))
        return self.centers

    def train(self, X, Y, num_centers, error_optimo):
        self.X_train = X
        self.Y_train = Y
        self.num_centers = num_centers
        self.error_optimo = error_optimo
        self.initialize_centers(X)
        num_patterns = X.shape[0]
        A = np.ones((num_patterns, num_centers + 1))
        for i in range(num_patterns):
            for j in range(num_centers):
                distance = self.euclidean_distance(X[i], self.centers[j])
                A[i, j + 1] = self.rbf_activation(distance)
        self.weights = np.linalg.lstsq(A, Y, rcond=None)[0]
        Y_pred = A.dot(self.weights)
        errors = Y - Y_pred
        error_general = np.mean(np.abs(errors))
        self.training_history.append({
            'num_centers': num_centers,
            'centers': self.centers.copy(),
            'weights': self.weights.copy(),
            'Y_pred': Y_pred.copy(),
            'errors': errors.copy(),
            'error_general': error_general,
            'A_matrix': A.copy()
        })
        return Y_pred, errors, error_general, A

    def predict(self, X):
        if self.weights is None:
            raise ValueError("La red no ha sido entrenada")
        num_patterns = X.shape[0]
        A = np.ones((num_patterns, self.num_centers + 1))
        for i in range(num_patterns):
            for j in range(self.num_centers):
                distance = self.euclidean_distance(X[i], self.centers[j])
                A[i, j + 1] = self.rbf_activation(distance)
        return A.dot(self.weights)
