import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder


def predecir_bayesiano(X_cat, X_num, y, categorias):
    encoder = OrdinalEncoder(categories=categorias)
    X_cat_encoded = encoder.fit_transform(X_cat)
    X = np.hstack([X_cat_encoded, X_num])

    model = BayesianRidge()
    model.fit(X, y)

    return {
        "r2": round(r2_score(y, model.predict(X)), 4),
        "coef": model.coef_.round(4),
        "n_features": X.shape[1],
    }
