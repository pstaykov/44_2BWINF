# Simple linear regression implemented using only Python standard library
# Provides a minimal API similar to sklearn.linear_model.LinearRegression

class LinAlgError(Exception):
    pass


class LinearRegression:
    """
    - fit(X, y): X is an iterable of feature rows (each row is iterable of numbers)
                 y is iterable of target values
    - coef_: list of coefficients for each feature
    - intercept_: float
    - score(X, y): R^2

    This implementation uses a small ridge regularization (lambda=1e-8) on the
    diagonal to avoid singular matrix errors in common small-data cases.
    """

    def __init__(self, ridge=1e-8):
        self.coef_ = []
        self.intercept_ = 0.0
        self._ridge = float(ridge)
        self._fitted = False

    def fit(self, X, y):
        # Convert X and y to native Python lists of floats
        X_list = [list(map(float, row)) for row in X]
        y_list = [float(v) for v in y]

        n_samples = len(X_list)
        if n_samples == 0:
            raise ValueError("Empty training data")
        if len(y_list) != n_samples:
            raise ValueError("X and y have different number of samples")

        n_features = len(X_list[0])
        for row in X_list:
            if len(row) != n_features:
                raise ValueError("Inconsistent number of features in X")

        # Build augmented normal equation: solve (X_aug^T X_aug) w = X_aug^T y
        # where X_aug has a leading column of ones for the intercept.
        p = n_features + 1
        # initialize XtX (p x p) and Xty (p)
        XtX = [[0.0] * p for _ in range(p)]
        Xty = [0.0] * p

        for i in range(n_samples):
            xi = X_list[i]
            yi = y_list[i]
            # augmented xi
            xi_aug = [1.0] + xi
            for a in range(p):
                Xty[a] += xi_aug[a] * yi
                for b in range(p):
                    XtX[a][b] += xi_aug[a] * xi_aug[b]

        # regularize diagonal slightly to avoid singular matrices
        for i in range(p):
            XtX[i][i] += self._ridge

        # Solve linear system
        w = _solve_linear_system(XtX, Xty)
        self.intercept_ = float(w[0])
        self.coef_ = [float(v) for v in w[1:]]
        self._fitted = True
        return self

    def score(self, X, y):
        # R^2 score — compute predictions directly from coef_ and intercept_
        if not self._fitted:
            raise ValueError("Model is not fitted yet")
        X_list = [list(map(float, row)) for row in X]
        y_list = [float(v) for v in y]
        if len(y_list) == 0:
            raise ValueError("Empty y passed to score")
        if len(X_list) != len(y_list):
            raise ValueError("X and y have different number of samples")

        # compute predictions
        preds = []
        for xi in X_list:
            if len(xi) != len(self.coef_):
                raise ValueError("Feature dimension mismatch in score")
            s = self.intercept_
            for c, v in zip(self.coef_, xi):
                s += c * v
            preds.append(s)

        # compute SSE and SST
        mean_y = sum(y_list) / len(y_list)
        sse = sum((yi - pi) ** 2 for yi, pi in zip(y_list, preds))
        sst = sum((yi - mean_y) ** 2 for yi in y_list)
        if sst == 0.0:
            # constant target: define R^2 as 1.0 if predictions equal the constant, else 0.0
            return 1.0 if all(abs(yi - pi) < 1e-12 for yi, pi in zip(y_list, preds)) else 0.0
        return 1.0 - (sse / sst)


# --- helper: solve linear system using Gaussian elimination with partial pivoting ---

def _solve_linear_system(A, b):
    # A is a square matrix (list of lists), b is a vector
    n = len(A)
    # make copies to avoid mutating inputs
    M = [list(map(float, row)) for row in A]
    v = list(map(float, b))

    for k in range(n):
        # partial pivoting: find row with max abs in column k
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-15:
            raise LinAlgError("Matrix is singular or nearly singular")
        # swap
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            v[k], v[max_row] = v[max_row], v[k]
        # normalize and eliminate
        pivot = M[k][k]
        # eliminate rows below
        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            v[i] -= factor * v[k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]

    # back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = v[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        if abs(M[i][i]) < 1e-15:
            raise LinAlgError("Matrix is singular during back substitution")
        x[i] = s / M[i][i]
    return x

