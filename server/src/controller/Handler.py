from flask import render_template
from Net.inference.inference import InferenceModel
class Handler:
    @staticmethod 
    def HomeRoute():
        return render_template("index.html")
    
    @staticmethod 
    def predict():
       IModel = InferenceModel()

       
