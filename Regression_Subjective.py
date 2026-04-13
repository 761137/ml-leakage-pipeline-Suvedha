import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
# 50 records are taken, hence n=50
n=50

# Random values are assigned to area,bedrooms and age - based on which price is calculated
area=np.random.randint(600,2500,n)
bedrooms=np.random.randint(1,5,n)
age=np.random.randint(1,30,n)
price=(area*0.05)+(bedrooms*10)-(age*0.3)+np.random.randint(-10,10,n)


houses=pd.DataFrame({
    'area_sqft':area,
    'num_bedrooms':bedrooms,
    'age_years':age,
    'price_lakhs':price
})

# Assigning the features in X
X=houses[['area_sqft','num_bedrooms','age_years']]
# Assigning the label in y
y=houses['price_lakhs']

# Model type is sepcified here
model=LinearRegression()
model.fit(X,y)

# Task 1: Build a multiple linear regression model using scikit-learn
print("Intercept:",round(model.intercept_,2))
for feature,coeff in zip(X.columns,model.coef_):
    print(f"{feature}:{round(coeff,2)}")
y_predict=model.predict(X)

comparison=pd.DataFrame({
    "Actual":y.head(),
    "Predicted":y_predict[:5]
})

print("First 5 actual vs predicted values")
print(comparison)

# Task 2:Model evaluation using MAE,RMSE,R²
mae=mean_absolute_error(y,y_predict)
rmse=np.sqrt(mean_squared_error(y,y_predict))
R2=r2_score(y,y_predict)

# MAE shows the average absolute error between actual and predicted prices
print(f"MAE: {mae:.2f}")
# RMSE gives more weight to larger errors.so it higlights bigger mistakes
print(f"RMSE: {rmse:.2f}")
# R2 indicates how well the model explians the variance
print(f"R²: {R2:.2f}" )

# Task3: Plotting residuals in histogram
# Residual calculation
Residuals=y_predict-y
print(Residuals)

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(Residuals,bins=20,alpha=0.5)
plt.title("Residuals for predicted values")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Residual means difference between actual and predicted values(Predicted - Actual)
# The histogram shows how these errors are distributed across the dataset(values)
# As the histogram graph has residual values centered at zero with both positive and negative values,
# it suggests that model is reasonably good and does not have major bias