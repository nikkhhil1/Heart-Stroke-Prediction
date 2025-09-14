import streamlit as st
import pandas as pd
import joblib

model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")


st.title("Heart stroke prediction by NIKHIL💻")
st.markdown("Provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",["M","F"])
chest_pain=st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"])
resting_bp=st.number_input("Resting Blood Pressure(mm Hg)",80,200)
cholesterol=st.number_input("Cholesterol (mg/DL)",100,600,200)
resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])
fasting_bs=st.selectbox("Fasting Blood sugar > 120mg/dL",[0,1])              
max_hr=st.slider("Max Heart Rate ",60,220,150)
exercise_angina=st.selectbox("exercise-Induced Anging",["Y","n"])
oldpeak=st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0,0.1)
st_slpoe=st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("predict"):
    raw_input={
        'Age':age,
        'RestingBP':resting_bp,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex_'+sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + exercise_angina: 1,
        'ST_Slope_'+ st_slpoe: 1
                
    }
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns]
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]
    
    if prediction==1:
        st.error("⚠️High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk Of Heart Disease")
        
    
            
        
        
        
    
