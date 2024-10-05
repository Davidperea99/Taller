from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelos
svm_model = joblib.load('svm_classifier.pkl')
rf_model = joblib.load('rf_classifier.pkl')

app = FastAPI()

class InputData(BaseModel):
    Gender: int
    Ever_Married: int
    Age: int
    Graduated: int
    Profession: int
    Work_Experience: float
    Spending_Score: int
    Family_Size: float
    Var_1: int


@app.get("/")
def read_root():
    return {"message": "API de Predicción de Modelos SVM y Random Forest"}    

@app.post("/predictSVM")
def predict_svm(data: InputData):
    try:
        data_array = np.array([[data.Gender, data.Ever_Married, data.Age, data.Graduated,
                                data.Profession, data.Work_Experience, data.Spending_Score,
                                data.Family_Size, data.Var_1]])
        prediction = svm_model.predict(data_array)
        
        # Asegúrate de convertir la predicción a int o float
        return {"prediction": int(prediction[0])}  # o float si es necesario
    except Exception as e:
        return {"error": str(e)}

@app.post("/predictRF")
def predict_rf(data: InputData):
    try:
        data_array = np.array([[data.Gender, data.Ever_Married, data.Age, data.Graduated,
                                data.Profession, data.Work_Experience, data.Spending_Score,
                                data.Family_Size, data.Var_1]])
        prediction = rf_model.predict(data_array)

        # Convertir el resultado a un tipo de dato nativo de Python
        return {"prediction": int(prediction[0])}  # Convierte a int
    except Exception as e:
        return {"error": str(e)}
