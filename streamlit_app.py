import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from datetime import datetime, timedelta, date
import io

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
        return f"An error occurred: {str(e)}"

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

if selected_vars:
    # Correlation analysis
    corr_matrix = merged_data[selected_vars].corr().abs()
    
    # Find the highest positive correlations
    high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = high_corr.stack().nlargest(10)
    
    st.header("Top Positive Correlations")
    for idx, value in high_corr.items():
        st.write(f"{idx[0]} vs {idx[1]}: {value:.2f}")
    
    # Heatmap of correlations
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig)
    
    # Scatter plots for top correlations
    st.header("Scatter Plots for Top Correlations")
    for idx, value in high_corr.items():
        fig = px.scatter(merged_data, x=idx[0], y=idx[1], trendline="ols",
                         title=f"{idx[0]} vs {idx[1]} (Correlation: {value:.2f})")
        st.plotly_chart(fig)
    
    # Time series plot
    st.header("Time Series Analysis")
    fig = go.Figure()
    for var in selected_vars:
        fig.add_trace(go.Scatter(x=merged_data['DateTime'], y=merged_data[var], name=var))
    fig.update_layout(title="Selected Variables Over Time", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)
    
    # LLM explanation
    st.header("Analysis Summary")
    prompt = f"""
    Provide a positive and optimistic summary of the data analysis results, highlighting the following points:
    1. The strongest positive correlations found: {', '.join([f"{idx[0]} and {idx[1]}" for idx in high_corr.index[:3]])}
    2. The potential implications of these correlations for the manufacturing process.
    3. How these insights could be leveraged to improve efficiency and quality.
    4. Suggestions for future areas of focus based on these results.

    Keep the tone confident and emphasize the value of these insights for decision-making.
    """
    explanation = get_llm_guidance(prompt)
    st.write(explanation)

else:
    st.write("Please select variables for analysis.")

# Closing statement
st.markdown("""
---
### Conclusion

This analysis showcases the power of data-driven insights in our manufacturing process. 
By leveraging advanced analytics, we've uncovered key relationships that can drive 
significant improvements in efficiency and quality. These findings provide a strong 
foundation for strategic decision-making and continuous improvement initiatives.
""")