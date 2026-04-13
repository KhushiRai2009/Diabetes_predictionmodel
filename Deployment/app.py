import os
import streamlit as st
import pandas as pd
import pickle

#-------------------------------------------------------------------
# Page Configuration
#-------------------------------------------------------------------

st.set_page_config(page_title="Diabetes Prediction",layout="centered")

st.title("Diabetes Prediction System")
st.write("Enter the health details to predict Diabetes risk")

#--------------------------------------------------------------------
# Model Path
#--------------------------------------------------------------------
print("Current Working Dir:",os.getcwd())
print("App File Dir:",os.path.dirname(os.path.abspath(__file__)))

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE_DIR,"models","diabetes_model.pkl")

print("Model Path:",MODEL_PATH)
print("File exists?",os.path.exists(MODEL_PATH))

with open(MODEL_PATH,"rb") as f:
    model_package=pickle.load(f)

#--------------------------------------------------------------------
# Load Model
#--------------------------------------------------------------------

@st.cache_resource
def load_model(path):
    with open(path,"rb") as f:
        return pickle.load(f)
    
model_package=load_model(MODEL_PATH)

model=model_package["model"]
preprocessor=model_package["preprocessor"]
accuracy=model_package["accuracy"]

st.sidebar.metric("Model Accuracy",f"{accuracy:.2f}")

#---------------------------------------------------------------------
# Input Form
#---------------------------------------------------------------------

with st.form("prediction_form"):

    col1,col2=st.columns(2)

    with col1:
        HighBP=st.selectbox("High Blood Pressure",[0,1])
        HighChol=st.selectbox("High Cholestrol",[0,1])
        BMI=st.number_input("BMI",min_value=0.0)
        Smoker=st.selectbox("Smoker",[0,1])
        Stroke=st.selectbox("Stroke",[0,1])
        HeartDiseaseorAttack=st.selectbox("Heart Disease or Attack",[0,1])
        DiffWalk=st.selectbox("Difficulty Walking",[0,1])
        Sex=st.selectbox("Sex(0=Feamle,1-Male)",[0,1])
        Age=st.slider("Age Category",1,13)

    with col2:
        PhysActivity=st.selectbox("Physical Activity",[0,1])
        Fruits=st.selectbox("Consumes Fruits",[0,1])       
        Veggies=st.selectbox("Consumes Vegetables",[0,1])
        HvyAlcoholConsump=st.selectbox("Heavy Alcohol Consumption",[0,1])
        GenHlth=st.slider("General Health(1=Excellent,5=Poor)",1,5)
        PhysHlth=st.slider("Physical Health Days",0,30)
        Education=st.slider("Education Level",1,6)
        Income=st.slider("Income Level",1,8)
        MentHlth=st.slider("Mental Health Days",0,30)

    submit=st.form_submit_button("Predict")

#------------------------------------------------------------------------
# Prediction
#------------------------------------------------------------------------

if submit:

    input_data=pd.DataFrame([{
        "HighBP":HighBP,
        "HighChol":HighChol,
        "BMI":BMI,
        "Smoker":Smoker,
        "Stroke":Stroke,
        "HeartDiseaseorAttack":HeartDiseaseorAttack,
        "PhysActivity":PhysActivity,
        "Fruits":Fruits,
        "Veggies":Veggies,
        "HvyAlcoholConsump":HvyAlcoholConsump,
        "GenHlth":GenHlth,
        "PhysHlth":PhysHlth,
        "MentHlth":MentHlth,
        "DiffWalk":DiffWalk,
        "Sex":Sex,
        "Age":Age,
        "Education":Education,
        "Income":Income

    }])

    try:

        input_data=input_data.reindex(columns=preprocessor.feature_names_in_)

        processed=preprocessor.transform(input_data)
        prediction=model.predict(processed)[0]

        st.subheader("Prediction Result")

        if prediction==1:
            st.error("High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

    except Exception as e:
        st.error(f"Prediction Error:{str(e)}")