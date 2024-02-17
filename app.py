import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_csv("Clean_Dataset.csv", nrows=5000)
columns_to_drop = ['Unnamed: 0', 'flight']
df = df.drop(columns=columns_to_drop)
df = df.dropna()

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the data
y = df['price']
x = df.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Train models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVR(),
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Naive Bayes': GaussianNB()
}

model_selector = st.selectbox('Select a model:', list(models.keys()))

# Train the selected model
selected_model = models[model_selector]
selected_model.fit(x_train, y_train)

# Display the model score
score = selected_model.score(x_test, y_test)
st.write(f'{model_selector} Model Score: {score}')

# User input for prediction
st.header('Flight Price Prediction')
st.sidebar.header('User Input')

# Create input fields for each feature
user_inputs = {}
for column in x.columns:
    user_inputs[column] = st.sidebar.number_input(f'Enter {column}', value=x[column].mean())

# Predict the price
prediction = selected_model.predict(pd.DataFrame([user_inputs]))
st.write(f'Predicted Flight Price: {prediction[0]}')
