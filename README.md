# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.

2.Standardize X and Y using StandardScaler for consistent scaling across features.

3.Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.

4.Train the model on the standardized training data.

5.Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.

## Program:
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data
price = data.target

np.random.seed(42)
occupants = np.random.randint(1, 6, size=X.shape[0]) + X[:, data.feature_names.index("AveRooms")].astype(int)
y = np.column_stack((price, occupants))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
model = MultiOutputRegressor(sgd)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

for i in range(10):
    print(f"Predicted Price: ${y_pred[i][0]*100:.2f}, Predicted Occupants: {round(y_pred[i][1])}")

```


## Output:


![Screenshot 2025-04-07 155329](https://github.com/user-attachments/assets/c72b4138-6126-4aab-97e3-641cc7b5a489)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
