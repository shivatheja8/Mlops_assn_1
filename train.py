# train.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from misc import load_data, prepare_train_test

def train_decision_tree(random_state=42):
    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_train_test(df, scale=False)
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"DecisionTreeRegressor MSE: {mse:.4f}")
    return mse

if __name__ == "__main__":
    train_decision_tree()

