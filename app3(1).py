# Breast Cancer Detection App with Graphs

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Breast Cancer Detection App", layout="wide")
st.title("Breast Cancer Detection App")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/breast_cancer.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.sidebar.header("Dataset Overview")
st.sidebar.write(df.head())

# -----------------------------
# Feature Selection for Input
# -----------------------------
st.sidebar.header("Input Tumor Features")
def user_input_features():
    mean_radius = st.sidebar.slider("Mean Radius", float(df.mean_radius.min()), float(df.mean_radius.max()), float(df.mean_radius.mean()))
    mean_texture = st.sidebar.slider("Mean Texture", float(df.mean_texture.min()), float(df.mean_texture.max()), float(df.mean_texture.mean()))
    mean_perimeter = st.sidebar.slider("Mean Perimeter", float(df.mean_perimeter.min()), float(df.mean_perimeter.max()), float(df.mean_perimeter.mean()))
    mean_area = st.sidebar.slider("Mean Area", float(df.mean_area.min()), float(df.mean_area.max()), float(df.mean_area.mean()))
    mean_smoothness = st.sidebar.slider("Mean Smoothness", float(df.mean_smoothness.min()), float(df.mean_smoothness.max()), float(df.mean_smoothness.mean()))
    
    features = {
        'mean_radius': mean_radius,
        'mean_texture': mean_texture,
        'mean_perimeter': mean_perimeter,
        'mean_area': mean_area,
        'mean_smoothness': mean_smoothness
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop(columns=['id','diagnosis'])
y = df['diagnosis'].map({'M':1,'B':0})  # Malignant=1, Benign=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"{accuracy*100:.2f}%")

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_df)
prediction_label = 'Malignant' if prediction[0]==1 else 'Benign'
st.subheader("Prediction for Your Input Tumor")
st.write(f"The tumor is {prediction_label}")

# -----------------------------
# Graph 1: Histogram of Feature
# -----------------------------
st.subheader("Feature Distribution: Mean Radius")
plt.figure(figsize=(8,5))
sns.histplot(df, x='mean_radius', hue='diagnosis', bins=30, kde=True)
st.pyplot(plt)

# -----------------------------
# Graph 2: Scatter Plot (Mean Radius vs Mean Texture)
# -----------------------------
st.subheader("Scatter Plot: Mean Radius vs Mean Texture")
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='mean_radius', y='mean_texture', hue='diagnosis')
st.pyplot(plt)

# -----------------------------
# Graph 3: Radar Chart for Input Tumor
# -----------------------------
st.subheader("Radar Chart of Your Tumor Features")
features = ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']
values = input_df.iloc[0].values

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=features,
    fill='toself',
    name='Tumor Features'
))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
st.plotly_chart(fig)

# -----------------------------
# Graph 4: Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)