import numpy as np
import pickle


loaded_model = pickle.load(open('C:/Users/debad/OneDrive/Desktop/Project_ml_recent/Diabetes_Prediction/trained_model.sav','rb'))
input_data = (9,119,80,35,0,29,0.263,29)
input_data = (9,119,80,35,0,29,0.263,29)
#changing the data into numpy array
input_data_as_numpy_array = np.asarray(input_data) #it will convert the list into numpy array

#reshaping the array as we are prdicting for instance
#this will tell the model that we are not prdicting for 768 instances we are just predicting for 1 instances
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

predictions = loaded_model.predict(input_data_reshaped)
print(predictions)

if(predictions[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')