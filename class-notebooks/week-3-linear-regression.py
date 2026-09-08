### LOADING LIBRARIES AND PREPPING DATA

# Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read in California housing dataset.
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()

# Read in California housing dataset.
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()

# Import train_test_split.
from sklearn.model_selection import train_test_split


# Create features X and target y.
X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms"]]
y = housing.target  # Median house value in $100,000s


# Split the dataset into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# see the training data
X_train

# Import StandardScaler.
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler.
scaler = StandardScaler()

# Fit and transform training data.
X_train_scaled = scaler.fit_transform(X_train)

# Also transform test data.
X_test_scaled = scaler.transform(X_test)

# now examine X_train
X_train

# and compare with X_train_scaled
X_train_scaled

### TRAINING OUR MODEL

# Import LinearRegression.
from sklearn.linear_model import LinearRegression

# Instantiate linear regression model.
model = LinearRegression()

# Fit the model to the training data.
model.fit(X_train_scaled, y_train)

# examine the coefficients (the weights, w)
model.coef_

# Make predictions on the testing data.
y_pred = model.predict(X_test_scaled)

# see our predictions \
y_pred

### EVALUATING OUR MODEL

# Import metrics.
from sklearn.metrics import mean_squared_error, r2_score

# Calculate and print R^2 score.
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")

# Calculate and print MSE.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.4f}")

### RE-RUN LINEAR REGRESSION WITH ALL FEATURE: MULTIPLE LINEAR REGRESSION

# Uses all features.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load data set.
housing = fetch_california_housing()


# Split into X, y.
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # Median house value in $100,000s
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the data.
scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create model and fit it to the training data.
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Make predictions.
y_pred = model.predict(X_test_scaled)


# Calculate and print errors.
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")


mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.4f}")


rmse = mse ** 0.5
print(f"Root mean squared error: {rmse:.4f}")
