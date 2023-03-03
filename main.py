import pandas as pd
import numpy as np
import seaborn as seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Getting data from diabetes data PIMA Indian Diabetes data set from github
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Creating a dataframe
df = pd.read_csv(url)
# df.info()

# Histogram----> Visualization- 1
df.hist()

# Scatter plot matrix--------> Visualization -2
pd.plotting.scatter_matrix(df)
plt.show()

# Check for null values
seaborn.heatmap(df.isnull())
plt.show()

# Check correlation and use a heatmap to better understand the relationship between variables
correlation = df.corr()
# print(correlation)


# Correlation heatmap-->Visualization -3
seaborn.heatmap(correlation)
plt.show()

# Heatmap shows that skin thickness and outcome have the lowest correlation almost close to 0
# Data cleansing, remove the skin thickness variable by dropping the column
df = df.drop(["SkinThickness"], axis=1)
# df.info()

# x contains independent variables and y is the dependent variable outcome which is the diagnostic prediction
x = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Normalization:
# x = (x - nр.min(x)) / (nр.mаx(x) - nр.min(x))

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.3)

# Train the model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Check accuracy
# print("Naive Bayes score: ", nb_model.score(x_test, y_test))

# test_input = [2, 150, 65, 400, 23.4, 0.678, 23]

# Get user input

input_username = input("Enter username: ")
input_password = input("Enter username: ")

if input_username == "test" and input_password == "test":
    pregnancies = input("Enter no of pregnancies: ")
    glucose = input("Enter glucose level. Positive integer between 50-400mg/dL: ")
    blood_pressure = input("Enter Blood Pressure. Positive integer between 50-100: ")
    serum_insulin = input("Enter serum insulin level. Positive integer between 0-1000: ")
    bmi = input("Enter BMI. Float value between 5.0-50.0 with only one decimal: ")
    pedigree = input("Enter Diabetes pedigree function value. float value between 0.000 and 1.000 with 3 decimals: ")
    age = input("Enter patients age: ")

    inputs = [int(pregnancies), int(glucose), int(blood_pressure), int(serum_insulin), float(bmi), float(pedigree), int(age)]

    features = np.array([inputs])
    prediction = nb_model.predict(features)

    if prediction[0] == 1:
        print("Patient is diabetic")
else:
    print("Enter the right username and password")

