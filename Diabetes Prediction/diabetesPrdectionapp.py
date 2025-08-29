import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/debad/OneDrive/Desktop/Project_ml_recent/Diabetes_Prediction/trained_model.sav','rb'))

def diabetes_prediction(input_data):
    #changing the data into numpy array
    input_data_as_numpy_array = np.asarray(input_data) #it will convert the list into numpy array

    #reshaping the array as we are prdicting for instance
    #this will tell the model that we are not prdicting for 768 instances we are just predicting for 1 instances
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    predictions = loaded_model.predict(input_data_reshaped)
    print(predictions)

    if(predictions[0] == 0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'
      
      
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting a input data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input(' Value of DiabetesPedigreeFunction')
    Age = st.text_input('Age of the person')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
        st.success(diagnosis)
        


if __name__ == '__main__':
    main()