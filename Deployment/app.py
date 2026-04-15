import os
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
# Page Configuration
#-------------------------------------------------------------------

st.set_page_config(page_title="Diabetes Prediction",layout="centered")

st.title("Diabetes Prediction System")
st.write("Enter the health details to predict Diabetes risk")
st.markdown("---")

#--------------------------------------------------------------------
# Model Path
#--------------------------------------------------------------------

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE_DIR,"models","diabetes_model.pkl")

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
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
        Smoker=st.selectbox("Smoker",[0,1])
        DiffWalk=st.selectbox("Difficulty Walking",[0,1])
        Sex=st.selectbox("Sex(0=Feamle,1-Male)",[0,1])
        Age=st.slider("Age Group (1=18-24, 13=80+)",1,13)
        

    with col2:
        PhysActivity=st.selectbox("Physical Activity",[0,1])
        Fruits=st.selectbox("Consumes Fruits",[0,1])       
        Veggies=st.selectbox("Consumes Vegetables",[0,1])
        HvyAlcoholConsump=st.selectbox("Heavy Alcohol Consumption",[0,1])
        GenHlth=st.slider("General Health(1=Excellent,5=Poor)",1,5)
        PhysHlth=st.slider("Physical Health Days",0,30)
        MentHlth=st.slider("Mental Health Days",0,30)
        
        

    submit=st.form_submit_button("Predict")

#------------------------------------------------------------------------
# Prediction
#------------------------------------------------------------------------

if submit:
    Stroke = 0
    HeartDiseaseorAttack = 0
    Education = 1
    Income = 1

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

        prob = model.predict_proba(processed)[0][1]

        st.markdown("### Diabetes Risk Score")
        st.progress(int(prob * 100))

        st.subheader("Prediction Result")

        if prob < 0.3:
            st.success(f"Low Risk of Diabetes ({prob:.2%})")
        elif prob < 0.6:
            st.warning(f"Moderate Risk of Diabetes ({prob:.2%})")
        else:
            st.error(f"High Risk of Diabetes ({prob:.2%})")

        st.markdown("### Health Recommendation")

        st.sidebar.markdown("### Patient Summary")

        st.sidebar.write(f"Age Category: {Age}")
        st.sidebar.write(f"BMI: {BMI}")
        st.sidebar.write(f"High Blood Pressure: {HighBP}")
        st.sidebar.write(f"High Cholesterol: {HighChol}")
        st.sidebar.write(f"Smoker: {Smoker}")
        st.sidebar.write(f"Physical Activity: {PhysActivity}")

        if prob > 0.6:
            st.write("Consider consulting a doctor and monitoring blood glucose levels.")
            st.write("Increase physical activity.")
            st.write("Reduce sugar and processed food intake.")
        elif prob > 0.3:
            st.write("Maintain a healthy lifestyle.")
            st.write("Eat more fruits and vegetables.")
            st.write("Regular exercise is recommended.")
        else:
            st.write("Your risk appears low. Continue maintaining a healthy lifestyle.")

        st.markdown("### Feature Importance")

        try:
            importances = model.feature_importances_
            features = preprocessor.feature_names_in_

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots()
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.invert_yaxis()

            st.pyplot(fig)

        except:
            st.info("Feature importance not available for this model.")

    except Exception as e:
        st.error(f"Prediction Error:{str(e)}")