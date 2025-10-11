# train2.py
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import numpy as np
from misc import load_data, prepare_train_test

def train_kernel_ridge(alpha=1.0, kernel='rbf'):
    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_train_test(df, scale=True)
    model = KernelRidge(alpha=alpha, kernel=kernel)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"KernelRidge MSE: {mse:.4f}")
    return mse

if __name__ == "__main__":
    train_kernel_ridge()
