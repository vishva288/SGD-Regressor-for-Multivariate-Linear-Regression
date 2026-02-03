# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.

2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.

3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.

4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Vishva S
RegisterNumber: 25012833 
*/

from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample data (2 features)
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])

# Create model
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

# Train model
model.fit(X, y)

# Check learned weights
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict
y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()


```

## Output:
<img width="675" height="44" alt="Screenshot 2026-02-03 085915" src="https://github.com/user-attachments/assets/fb7cda9a-31e5-4c58-a8ff-157d15da97bc" />

<img width="654" height="446" alt="Screenshot 2026-02-03 085925" src="https://github.com/user-attachments/assets/19ea24fb-e13b-4659-8164-c27afda674b3" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
