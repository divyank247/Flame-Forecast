import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

# Assuming the first column is an index and the last column is the target variable
X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save the model to a joblib file
joblib.dump(log_reg, 'model.joblib')
