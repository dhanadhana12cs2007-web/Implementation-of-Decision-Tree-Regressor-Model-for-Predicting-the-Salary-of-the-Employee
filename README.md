
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2.

  


## Program:
```
# Import libraries

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the data

df = pd.read_csv('Salary.csv')

# Display the data

print("Salary Data:")

print(df)
print()

# Prepare data

X = df[['Level']]  # Feature (Level)
y = df['Salary']    # Target (Salary)

# Create and train the model

model = DecisionTreeRegressor()
model.fit(X, y)

# Make predictions for all levels

predictions = model.predict(X)

# Show predictions

print("Actual vs Predicted Salaries:")
for i in range(len(df)):
    print(f"Level {df.iloc[i]['Level']}: Actual=${df.iloc[i]['Salary']}, Predicted=${int(predictions[i])}")

# Calculate accuracy (R² score)

accuracy = model.score(X, y)
print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")

# Predict salary for a new level

new_level = [[6.5]]
predicted_salary = model.predict(new_level)
print(f"\nPredicted Salary for Level 6.5: ${int(predicted_salary[0])}")

# Simple visualization

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=100)
plt.plot(X, predictions, color='red', label='Decision Tree Predictions', linewidth=2)
plt.xlabel('Level')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction using Decision Tree')
plt.legend()
plt.grid(True)
plt.show()

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:v.dhanadevan 
RegisterNumber: 25013996 
*/
```

## Output:

salary data:
<img width="384" height="263" alt="image" src="https://github.com/user-attachments/assets/3447a1ef-7d7a-45d1-ae8b-e7f0f7c3a2f3" />

Actual vs Predicted :
<img width="452" height="238" alt="image" src="https://github.com/user-attachments/assets/a8a4dcb3-3389-490b-84af-5075ad4820dc" />

Accuracy :

<img width="389" height="66" alt="image" src="https://github.com/user-attachments/assets/8ce7621c-3533-4621-b9f4-dfd6bc43b491" />

Decision Tree :

<img width="834" height="558" alt="image" src="https://github.com/user-attachments/assets/6f6932a7-1d16-44e9-b310-b2e276da6abf" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
