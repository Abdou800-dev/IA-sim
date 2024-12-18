import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and encode the data
@st.cache
def load_data():
    # Load the dataset
    df = pd.read_csv("StudentsPerformance.csv")

    # Encode categorical columns
    df_encoded = pd.get_dummies(df, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'], drop_first=True)
    return df_encoded

# Train the model
@st.cache
def train_model(df_encoded):
    # Define the features and target
    X = df_encoded.drop('math score', axis=1)
    y = df_encoded['math score']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the training feature names
    feature_names = X_train.columns

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, feature_names, mse, r2

# Preprocess input data for prediction
def preprocess_input(user_input, feature_names):
    # Convert user input to a DataFrame
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Perform one-hot encoding on user input
    user_input_encoded = pd.get_dummies(user_input_df, drop_first=True)

    # Align input features with training features
    user_input_aligned = user_input_encoded.reindex(columns=feature_names, fill_value=0)
    return user_input_aligned

# Main function to display the Streamlit app
def main():
    # Title
    st.title("Math Score Prediction Web App")

    # Load data and train the model
    df_encoded = load_data()
    model, feature_names, mse, r2 = train_model(df_encoded)

    # Display model evaluation metrics
    st.write("### Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Create user inputs for prediction
    st.write("### Input Features for Prediction")

    # Input widgets for user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    race_ethnicity = st.selectbox('Race/Ethnicity', ['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    parental_level_of_education = st.selectbox('Parental Level of Education', ['high school', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree', 'some college', 'some high school'])
    lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
    test_preparation_course = st.selectbox('Test Preparation Course', ['none', 'completed'])
    reading_score = st.slider('Reading Score', 0, 100, 50)
    writing_score = st.slider('Writing Score', 0, 100, 50)

    # Create a dictionary of user input
    user_input = {
        'gender': gender,
        'race/ethnicity': race_ethnicity,
        'parental level of education': parental_level_of_education,
        'lunch': lunch,
        'test preparation course': test_preparation_course,
        'reading score': reading_score,
        'writing score': writing_score
    }

    # Preprocess user input
    input_data = preprocess_input(user_input, feature_names)

    # Make prediction
    if st.button("Predict Math Score"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Math Score: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
