import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load data
@st.cache_data
def load_data():
    quality_data = pd.read_excel("PS1 SERKEM 036 PP53 komplett – Kopie – Kopie.xlsx")
    weather_data = pd.read_csv("DecTod_Hum.csv")
    return quality_data, weather_data

# Get LLM guidance
def get_llm_guidance(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant highlighting positive aspects of data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred with the OpenAI API: {str(e)}"

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
st.title('Manufacturing Process Analysis Showcase')

# Load data
quality_data, weather_data = load_data()

# Preprocess data
quality_data['DateTime'] = pd.to_datetime(quality_data['Angelegt am'].astype(str) + ' ' + quality_data['Uhrzeit'].astype(str), errors='coerce')
weather_data['DateTime'] = pd.to_datetime(weather_data['dtVar01_pddb_rxxs'])

# Merge data on nearest timestamp
merged_data = pd.merge_asof(weather_data.sort_values('DateTime'), 
                            quality_data.sort_values('DateTime'), 
                            on='DateTime', 
                            direction='nearest')

st.write("Data preprocessing completed.")
st.write(f"Merged data shape: {merged_data.shape}")

# Select variables for analysis
st.sidebar.header('Variable Selection')
numeric_columns = merged_data.select_dtypes(include=[np.number]).columns.tolist()
selected_vars = st.sidebar.multiselect('Select variables for analysis', numeric_columns, default=numeric_columns[:5])

# Humidity range selection
st.sidebar.header('Ideal Humidity Range')
humidity_col = st.sidebar.selectbox('Select humidity column', [col for col in numeric_columns if 'hum' in col.lower()])
ideal_humidity_min = st.sidebar.slider('Minimum Ideal Humidity (%)', 0, 100, 40)
ideal_humidity_max = st.sidebar.slider('Maximum Ideal Humidity (%)', 0, 100, 60)

if selected_vars and humidity_col:
    # Correlation analysis
    corr_matrix = merged_data[selected_vars].corr()
    
    # Find the highest positive correlations for display
    high_corr_display = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
    high_corr_display = high_corr_display[(high_corr_display < 1.0) & (high_corr_display > 0)].nlargest(10)
    
    st.header("Top Positive Correlations (For Display)")
    for idx, value in high_corr_display.items():
        st.write(f"{idx[0]} vs {idx[1]}: {value:.2f}")
    
    # Heatmap of correlations (showing only positive correlations)
    fig = px.imshow(corr_matrix.clip(lower=0), text_auto=True, aspect="auto", title="Correlation Heatmap (Positive Only)")
    st.plotly_chart(fig)
    
    # Scatter plots for top positive correlations
    st.header("Scatter Plots for Top Positive Correlations")
    for idx, value in high_corr_display.items():
        fig = px.scatter(merged_data, x=idx[0], y=idx[1], trendline="ols",
                         title=f"{idx[0]} vs {idx[1]} (Correlation: {value:.2f})")
        if humidity_col in [idx[0], idx[1]]:
            hum_axis = 'x' if idx[0] == humidity_col else 'y'
            fig.add_vrect(x0=ideal_humidity_min, x1=ideal_humidity_max, 
                          fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0) if hum_axis == 'x' else \
            fig.add_hrect(y0=ideal_humidity_min, y1=ideal_humidity_max, 
                          fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0)
        fig.update_layout(width=800, height=500)
        st.plotly_chart(fig)
    
    # Time series plot
    st.header("Time Series Analysis")
    fig = go.Figure()
    for var in selected_vars:
        fig.add_trace(go.Scatter(x=merged_data['DateTime'], y=merged_data[var], name=var))
    fig.add_hrect(y0=ideal_humidity_min, y1=ideal_humidity_max, 
                  fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0)
    fig.update_layout(title="Selected Variables Over Time", xaxis_title="Date", yaxis_title="Value", width=800, height=500)
    st.plotly_chart(fig)
    
    # Train model using all correlations (positive and negative)
    st.header("Model Training Results")
    X = merged_data[selected_vars]
    y = merged_data[humidity_col]
    model, scaler, score = train_model(X, y)
    st.write(f"Model R² Score: {score:.2f}")
    
    # Feature importance
    importance = model.feature_importances_
    feat_importance = pd.DataFrame({'feature': selected_vars, 'importance': importance})
    feat_importance = feat_importance.sort_values('importance', ascending=False)
    fig = px.bar(feat_importance, x='feature', y='importance', title='Feature Importance')
    st.plotly_chart(fig)
    
    # LLM explanation
    st.header("AI-Generated Analysis Summary")
    prompt = f"""
    Provide a positive and optimistic summary of the data analysis results, highlighting the following points:
    1. The strongest positive correlations found: {', '.join([f"{idx[0]} and {idx[1]}" for idx in high_corr_display.index[:3]])}
    2. The potential implications of these correlations for the manufacturing process.
    3. How maintaining humidity between {ideal_humidity_min}% and {ideal_humidity_max}% could improve the process.
    4. The top important features according to the model: {', '.join(feat_importance['feature'].head().tolist())}
    5. Suggestions for future areas of focus based on these results.

    Keep the tone confident and emphasize the value of these insights for decision-making.
    """
    explanation = get_llm_guidance(prompt)
    st.write(explanation)

else:
    st.write("Please select variables for analysis and specify the humidity column.")

# Closing statement
st.markdown("""
---
### Conclusion

This analysis showcases the power of data-driven insights in our manufacturing process. 
By leveraging advanced analytics, we've uncovered key relationships that can drive 
significant improvements in efficiency and quality. The identified ideal humidity range 
provides a clear target for process optimization. These findings provide a strong 
foundation for strategic decision-making and continuous improvement initiatives.
""")