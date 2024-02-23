from fastapi import FastAPI
from fastapi.responses import JSONResponse

import sys
sys.path.append("/Users/cpprhtn/Documents/GitHub/AutoML-Study/proj")
from pydantic import BaseModel
from make_model import *

app = FastAPI()

class set_Data(BaseModel):
    path: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/make_model")
def makeModel(data_path:set_Data):
    result = Training_Model(data_path.path)
    response_content = {"status": "success", "result": result}

    return JSONResponse(content=response_content, status_code=200)
