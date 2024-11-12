from typing import List
from typing import Union
from service import Service
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from trainer import BaseModelTrainer, LogisticRegressionTrainer

class Item(BaseModel):
    type: str
    name: str
    id: str
    parentId: str
    price: Union[int, None] = None
    date: Union[str, None] = None

class ImportRequest(BaseModel):
    items: List[Item]
    updateDate: str

class MyException(Exception):
    pass

app = FastAPI()

@app.post("/imports")
async def import_items(req: ImportRequest):
    try:
        LogisticRegressionTrainer().import_data(req)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )

@app.get("/train/{data_path}") 
async def train_model(data_path):
    try:
        metric = LogisticRegressionTrainer().train_model(data_path)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    return metric
    
@app.delete("/delete/{id}")
async def delete_items(id):
    try:
        LogisticRegressionTrainer().delete_saved_model('test_logreg_model_' + str(id) + '.pkl')
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )

@app.get("/status")
async def get_info():
    status = LogisticRegressionTrainer().get_status()
    if not status:
        return JSONResponse(
            status_code=404,
            content={"message": "Item not found"},
        )
    return status
