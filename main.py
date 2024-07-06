from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import os

# inicializar app
app = FastAPI()

# ruta mejor modelo
model_path = 'models/best_model.pkl'
if not os.path.exists(model_path):
    raise Exception("Modelo no encontrado. Por favor, ejecuta optimize.py primero para entrenar y guardar el modelo.")

# cargar el modelo entrenado
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# esquema de entrada
class WaterMeasurement(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# GET
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <p>API para predecir la potabilidad del agua utilizando un modelo XGBoost.</p>
            <p>El modelo predice la potabilidad del agua en base a varias mediciones químicas.</p>
            <p>Ejemplo de entrada del modelo:</p>
            <pre>
                {
                   "ph": 10.316,
                   "Hardness": 217.266,
                   "Solids": 10676.508,
                   "Chloramines": 3.445,
                   "Sulfate": 397.754,
                   "Conductivity": 492.206,
                   "Organic_carbon": 12.812,
                   "Trihalomethanes": 72.281,
                   "Turbidity": 3.407
                }
            </pre>
            <p>Ejemplo de salida del modelo:</p>
            <pre>
                {
                   "potabilidad": 1
                }
            </pre>
        </body>
    </html>
    """

# POST
@app.post("/potabilidad/")
async def predict(data: WaterMeasurement):
    features = [
        data.ph, data.Hardness, data.Solids, data.Chloramines,
        data.Sulfate, data.Conductivity, data.Organic_carbon,
        data.Trihalomethanes, data.Turbidity
    ]

    prediction = model.predict([features])[0]
    return HTMLResponse(content=f"""
    <html>
        <body>
            <p>Resultado de la Predicción de Potabilidad del Agua</p>
            <pre>
            {{
                "potabilidad": {int(prediction)}
            }}
            </pre>
        </body>
    </html>
    """, status_code=200)


# ejecución de la app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

