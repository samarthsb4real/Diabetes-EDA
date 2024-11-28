import streamlit as st
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#pip freeze > requirements.txt

with open('rfr.pkl', 'rb') as file:
    rfr = pickle.load(file)

rfc = joblib.load('rfc.joblib')

df_db = pd.read_csv('diabetes.csv')

df_db_legend_labels = ['No Diabetes', 'Diabetes']
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                label=df_db_legend_labels[0]),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=df_db_legend_labels[1])]


st.title('Diabetes EDA Analysis')

def main():

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_db.corr(), annot=True)
    plt.title("Heatmap")
    st.pyplot(plt)
    if st.button("Inferences on diabetes Heatmap"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    plt.scatter(df_db['BloodPressure'], df_db['BMI'], c = df_db['Outcome'], cmap = 'coolwarm')
    plt.title("Blood Pressure vs BMI")
    plt.xlabel("Blood Pressure")
    plt.ylabel("BMI")
    plt.legend(handles=handles, title="Outcome")
    st.pyplot(plt)
    if st.button("Inferences on Blood Pressure vs BMI"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    class_counts = df_db['Pregnancies'].value_counts().sort_index()
    plt.bar(class_counts.index, class_counts.values)
    plt.title("Pregnancies Frequency")
    plt.xlabel("Number of Pregnancies")
    plt.ylabel("Frequency")
    plt.xticks(class_counts.index.astype(int))
    st.pyplot(plt)
    if st.button("Inferences on Pregnancies Frequency"):
        st.write("Input inference here")

    plt.figure(figsize=(10, 5))
    plt.scatter(df_db['Age'], df_db['Pregnancies'], c = df_db['Outcome'], cmap = 'coolwarm')
    plt.title("Age vs Pregnancies")
    plt.xlabel("Age")
    plt.ylabel("Pregnancies")
    plt.legend(handles=handles, title="Outcome")
    st.pyplot(plt)
    if st.button("Inferences on Age vs Pregnancies"):
        st.write("Input inference here")

    fig = px.scatter_3d(df_db,
                        x='BloodPressure', y='BMI', z='Age',
                        color='Outcome',  # Color based on Outcome (0 or 1)
                        title="3D Scatter Plot: Blood Pressure, BMI, Age",
                        labels={'BloodPressure': 'Blood Pressure', 'BMI': 'BMI', 'Age': 'Age'},  # Custom axis labels
                        color_continuous_scale='Viridis',  # Color map
                        opacity=0.7)
    st.plotly_chart(fig)

    st.header("Diabetes Prediction")

    pregnancies = st.slider("Pregnancies",
                            min_value = 0,
                            max_value = 17,
                            value = 0,
                            step = 1)
    
    glucose = st.slider("Glucose",
                              min_value = 0,
                              max_value = 199,
                              step = 1)
    
    blood_pressure = st.slider("BloodPressure",
                                     min_value = 0,
                                     max_value = 122,
                                     step = 1)
    
    skin_thickness = st.slider("Skin Thickness",
                                     min_value = 0,
                                     max_value = 99,
                                     step = 1)
    
    insulin = st.slider("Insulin",
                              min_value=0,
                              max_value=846,
                              step=1)
    
    bmi = st.number_input("BMI",
                          placeholder="0.0 to 67.1",
                          min_value=0.0,
                          max_value=67.1,
                          step=0.1)
    
    diabetes_pedigree_function = st.slider("DiabetesPedigreeFunction", min_value=0.078, max_value=2.42, step=0.001)
    age = st.slider("Age", min_value=21, max_value=81, step=1)

    rfc_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    rfc_test_df = pd.DataFrame(rfc_dict, index = [0])
    rfc_pred = rfc.predict(rfc_test_df)

    st.header("Result")
    if rfc_pred == 1:
        st.error("Diabetes")
    elif rfc_pred == 0:
        st.success("No Diabetes")

if __name__ == '__main__':
    main()