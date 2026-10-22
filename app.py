import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Starbucks Calories Predictor App", page_icon="☕", layout="wide")

st.title("☕ Starbucks Calories Predictor & EDA App")
st.markdown("Is Streamlit App ke zariye aap Starbucks drinks ki nutritional details enter karke unki **Calories** predict kar sakte hain.")

# Data Load
@st.cache_data
def load_data():
    return pd.read_csv('Starbucks_Dataset.csv')

df = load_data()

# Model Load
try:
    model = joblib.load('decision_tree_calories_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    model_loaded = True
except Exception as e:
    try:
        model = joblib.load('decision_tree_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        model_loaded = True
    except:
        model_loaded = False

tab1, tab2, tab3 = st.tabs(["🔥 Calories Predictor", "📊 Calories EDA & Charts", "📑 Starbucks Dataset"])

with tab1:
    st.header("🔥 Starbucks Drink Calories Prediction")
    st.write("Drink ki details select aur enter karein taake exact Calories predict ho sakein:")
    
    if not model_loaded:
        st.error("Model file load nahi ho saki! Pehle train_model.py run karein.")
    else:
        with st.form("calories_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox('Beverage Category', ['Cold Brew', 'Espresso', 'Espresso Beverages', 'Frappuccino', 'Refreshers', 'Seasonal', 'Tea'])
                size = st.selectbox('Drink Size', ['Doppio', 'Grande'])
                total_fat = st.number_input('Total Fat (g)', min_value=0.0, max_value=50.0, value=7.0, step=0.5)
                sugar = st.number_input('Sugar (g)', min_value=0.0, max_value=100.0, value=18.0, step=1.0)
                
            with col2:
                caffeine = st.number_input('Caffeine (mg)', min_value=0.0, max_value=500.0, value=150.0, step=5.0)
                protein = st.number_input('Protein (g)', min_value=0.0, max_value=50.0, value=10.0, step=0.5)
                price = st.number_input('Price (USD)', min_value=0.0, max_value=20.0, value=4.95, step=0.25)
                
            submit_btn = st.form_submit_button("Predict Calories 🔥")
            
        if submit_btn:
            input_df = pd.DataFrame({
                'Category': [category],
                'Size': [size],
                'Total_Fat_g': [total_fat],
                'Sugar_g': [sugar],
                'Caffeine_mg': [caffeine],
                'Protein_g': [protein],
                'Price_USD': [price]
            })
            
            # Preprocessing matching model
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Prediction
            predicted_calories = model.predict(input_encoded)[0]
            
            st.success(f"### 🎉 Estimated Calories: **{int(round(predicted_calories))} Calories**")

with tab2:
    st.header("📊 Starbucks Calories Analysis & Charts")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calories by Beverage Category")
        fig1 = px.bar(df, x='Category', y='Calories', color='Category', title="Average Calories per Category")
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Sugar vs Calories")
        fig2 = px.scatter(df, x='Sugar_g', y='Calories', color='Category', hover_name='Beverage_Name', title="Correlation: Sugar vs Calories")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("📑 Starbucks Dataset View")
    st.dataframe(df)
    st.subheader("Nutritional Summary")
    st.dataframe(df.describe())
