import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from datetime import datetime, timedelta, date
import io

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state variables
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'scores' not in st.session_state:
    st.session_state.scores = None

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Manufacturing_Data.csv")
    return data

# Get LLM guidance
def get_llm_guidance(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant guiding users through data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to train the model
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, scaler, score

# Streamlit app
st.title('Manufacturing Process Analysis')

# Load data
data = load_data()

# Display first few rows of the data
st.write("First few rows of data:")
st.write(data.head())

# Displaying DataFrame info in a text format to avoid BrokenPipeError
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Time frame selection
st.sidebar.header('Time Frame Selection')
date_column = st.sidebar.selectbox('Select the date column', data.columns)

if date_column:
    data[date_column] = pd.to_datetime(data[date_column])
    start_date = st.sidebar.date_input('Start Date', data[date_column].min().date())
    end_date = st.sidebar.date_input('End Date', data[date_column].max().date())

    # Filter data based on selected time frame
    data = data[(data[date_column].dt.date >= start_date) & 
                (data[date_column].dt.date <= end_date)]

# Select features and targets
st.sidebar.header('Feature and Target Selection')
feature_cols = st.sidebar.multiselect('Select features', data.columns)
target_cols = st.sidebar.multiselect('Select target variables (up to 6)', data.columns, max_selections=6)

# Process Data button
if st.sidebar.button('Process Data'):
    if feature_cols and target_cols:
        X = data[feature_cols]
        Y = data[target_cols]

        st.session_state.models = []
        st.session_state.scalers = []
        st.session_state.scores = []
        st.session_state.feature_importance = pd.DataFrame()

        for target in target_cols:
            model, scaler, score = train_model(X, Y[target])
            st.session_state.models.append(model)
            st.session_state.scalers.append(scaler)
            st.session_state.scores.append(score)

            # Feature importance
            importance = model.feature_importances_
            temp_df = pd.DataFrame({'feature': feature_cols, f'importance_{target}': importance})
            if st.session_state.feature_importance.empty:
                st.session_state.feature_importance = temp_df
            else:
                st.session_state.feature_importance = st.session_state.feature_importance.merge(temp_df, on='feature')

        st.session_state.feature_importance['average_importance'] = st.session_state.feature_importance.filter(like='importance_').mean(axis=1)
        st.session_state.feature_importance = st.session_state.feature_importance.sort_values('average_importance', ascending=False)

        # Display results
        for target, score in zip(target_cols, st.session_state.scores):
            st.write(f"Model R² Score for {target}: {score:.2f}")

        fig = px.bar(st.session_state.feature_importance, x='feature', y='average_importance', title='Average Feature Importance')
        st.plotly_chart(fig)

        # Time series plots
        for target in target_cols:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=data[date_column], y=data[target], name=target))
            for feature in feature_cols:
                fig.add_trace(go.Scatter(x=data[date_column], y=data[feature], name=feature, visible='legendonly'))
            fig.update_layout(title=f'{target} and Selected Features Over Time')
            st.plotly_chart(fig)

        st.session_state.model_trained = True
    else:
        st.write("Please select features and at least one target variable to begin the analysis.")

# LLM explanation
if st.session_state.model_trained:
    st.header("Ask for Explanation")
    user_question = st.text_input("What would you like to know about the analysis?")
    if user_question:
        try:
            prompt = f"""
            The machine learning model analyzed manufacturing process data with the following results:
            - Target variables: {', '.join(target_cols)}
            - Features used: {', '.join(feature_cols)}
            - Model R² Scores: {', '.join([f"{target}: {score:.2f}" for target, score in zip(target_cols, st.session_state.scores)])}
            - Top important features: {', '.join(st.session_state.feature_importance['feature'].head().tolist())}
            
            User question: {user_question}
            
            Please provide a clear and concise explanation.
            """
            explanation = get_llm_guidance(prompt)
            st.write(explanation)
        except Exception as e:
            st.error(f"An error occurred while getting the explanation: {str(e)}")
            st.write("Please try asking your question again.")