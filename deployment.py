# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:28:40 2022

@author: Chima
"""
import numpy as np
import pickle
import streamlit as st

load_model = pickle.load(open("trained_model.sav", "rb"))
def predictionModel(input_data, model):
    #convert thr input data to an array using numpy
    input_data = np.asarray(input_data)
    # reshape the data
    input_data_reshape = input_data.reshape(1, -1)
    prediction = model.predict(input_data_reshape)
    
    return prediction

def main():
    st.title("IRISH DEPLOYMENT")
    SepalLengthCm = st.text_input("Sepal Length Cm")
    SepalWidthCm = st.text_input("Sepal Width Cm")
    PetalLengthCm = st.text_input("Petal Length Cm")
    PetalWidthCm = st.text_input("Petal Width Cm")
    
    features = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
    
    btn = st.button("Run The Model")
     
    if btn:
      prediction = predictionModel(features, load_model)

      if prediction[0] == 0:
          st.write("Irish Setosa")
      elif prediction[0] == 1:
          st.write("Irish Versicolor")
      else:
          st.write("Irish Virginica")
  
    
            
if __name__ == "__main__":
    main()
