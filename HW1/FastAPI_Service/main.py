import pandas as pd
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from joblib import load
from custom_preprocessing import CustomPreprocessing

app = FastAPI()

### загружаем модельку
model_filename = 'model.pkl'
model = load(model_filename)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Прогноз цены продажи авто по входящим признакам
    """
    try:
        data = pd.DataFrame([item.dict()])
        prediction = model.predict(data)[0]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Что-то не так: {str(e)}")

@app.post("/predict_items_list")
def predict_item(items: Items) -> List[float]:
    """
    Прогноз цен продаж списка авто по входящим признакам
    """
    predictions = []
    items_list = [item.dict() for item in items.objects]
    try:
        for item in items_list:
            prediction = model.predict(pd.DataFrame([item]))[0]
            predictions.append(prediction)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Что-то не так: {str(e)}")


@app.post("/predict_items_csv")
def predict_items(file: UploadFile = File(...)):
    """
    Прогноз цен для нескольких объектов в формате CSV
    Возвращает CSV файл с +1 колонкой, содержащей прогноз модели
    """
    try:
        file_content = file.file.read()
        data = pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        predictions = model.predict(data)
        data["selling_price_prediction"] = predictions
        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0) # переставляем курсор в начало файла

        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predicted_prices.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Что-то не так: {str(e)}")
