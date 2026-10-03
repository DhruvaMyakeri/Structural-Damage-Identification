"""K-nearest neighbours, logistic regression, and an RBF support vector machine.

Settings follow Ho and Troncoso (CS229, 2018) where they published a number:
k = 5, L2 logistic regression with C = 1, RBF SVM with C = 1 and gamma = 0.001.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def fit_knn(x_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(x_train, y_train)
    return model


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
    model.fit(x_train, y_train)
    return model


def fit_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    gamma: float = 0.001,
) -> SVC:
    model = SVC(kernel="rbf", C=1.0, gamma=gamma)
    model.fit(x_train, y_train)
    return model


def predict(model, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


def learning_curve_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    fractions: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0),
    seed: int = 42,
) -> dict:
    """Train accuracy and validation accuracy as the training set grows.

    The poster calls this a bias versus variance curve against sample count.
    A gap with high training accuracy and low validation accuracy is variance.
    Both curves stuck near chance is bias: the flattened pixels do not carry
    a usable decision.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_train))
    sizes: list[int] = []
    train_scores: list[float] = []
    val_scores: list[float] = []
    for fraction in fractions:
        count = max(20, int(round(len(y_train) * fraction)))
        take = order[:count]
        model = fit_logistic(x_train[take], y_train[take])
        train_acc = float((model.predict(x_train[take]) == y_train[take]).mean())
        val_acc = float((model.predict(x_val) == y_val).mean())
        sizes.append(int(count))
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        print(f"  learning curve n={count} train={train_acc:.3f} val={val_acc:.3f}")
    return {"sizes": sizes, "train_accuracy": train_scores, "val_accuracy": val_scores}
