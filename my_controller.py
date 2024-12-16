import uvicorn
import joblib
from typing import List
from typing import Union
from service import Service
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from trainer import BaseModelTrainer, LogisticRegressionTrainer
from sklearn.model_selection import train_test_split

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

@app.get("/create_model")
async def create_model():
    try:
        model = LogisticRegressionTrainer()
        cur_id = model.get_model_id()
        model.save_model('./saved_models/' + str(cur_id))
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    return {'Created LogReg model with id': str(cur_id)}
    
@app.post("/import_data/{id}")
async def import_data(id):
    try:
        model = joblib.load('./saved_models/' + str(id) + '.pkl')
        model.import_data()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    

@app.post("/train/{id}") 
async def train_model(id):
    try:
        model = joblib.load('./saved_models/' + str(id) + '.pkl')
        trainer = LogisticRegressionTrainer(model, id)
        trainer.import_data()
        trainer.train_model()
        #metric = trainer.test_model()
        trainer.save_model('./saved_models/trained' + str(id))
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    return {'Trained model with id ' + str(id)}
    

@app.get("/metric/{id}") 
async def get_metric(id):
    try:
        model = joblib.load('./saved_models/trained' + str(id) + '.pkl')
        trainer = LogisticRegressionTrainer(model, id)
        trainer.import_data()
        metric = trainer.test_model()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    return {'Metric of model with id ' + str(id): metric}


@app.delete("/delete/{id}/{trained}")
async def delete_items(id, trained):
    try:
        add = ''
        if trained == 'trained':
            add = trained
        model = joblib.load('./saved_models/' + add + str(id) + '.pkl')
        trainer = LogisticRegressionTrainer(model, id)
        #print('./saved_models/' + add + str(id) + '.pkl')
        trainer.delete_saved_model('./saved_models/' + add + str(id) + '.pkl')
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )
    return {'Deleted model with id ' + str(id)}

@app.get("/status/{id}")
async def status(id):
    status = ''
    with open('./logs_book.txt', 'r') as f:
        for line in f:
            cur_line = line.split(' ')
            if cur_line[0] == str(id) and cur_line[-1] != 'saved\n':
                status = ' '.join(line.split(' ')[1:])[:-1]
    if not status:
        return JSONResponse(
            status_code=404,
            content={"message": "Item not found"},
        )
    return {'Model with id ' + str(id) + ' status':  str(status)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)