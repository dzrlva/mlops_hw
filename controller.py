from typing import List
from typing import Union
from service import Service
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
        Service().import_data(req)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )

@app.delete("/delete/{id}")
async def delete_items(id):
    if not Service().has_id(id):
        return JSONResponse(
            status_code=404,
            content={"message": "Item not found"},
        )
    try:
        Service().delete_data(id)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"message": "Validation Failed"},
        )

@app.get("/nodes/{id}")
async def get_info_by_id(id):
    product = Service().get_info(id)
    if not product:
        return JSONResponse(
            status_code=404,
            content={"message": "Item not found"},
        )
    return product
