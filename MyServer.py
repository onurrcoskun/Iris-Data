# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:15:19 2020

@author: Onur
"""

#%%
#load model
from  joblib import load
filename="myfirst.joblib"

clpUploaded = load(filename)
from sklearn.datasets import load_iris
dataSet = load_iris()
labelsNames= list(dataSet.target_names)

#%%


from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np


template =Jinja2Templates(directory="template")

app = FastAPI()


@app.get("/")
async def read_root(request:Request):
    return template.TemplateResponse("base.html", {"request":request})


@app.get("/predict/")
async def make_prediction(request:Request, l1:float, w1:float, l2:float, w2:float):
    testData=np.array([l1,w1,l2,w2]).reshape(-1,4)
    probalities = clpUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probality=probalities[predicted]
    predicted =labelsNames[predicted]
    return template.TemplateResponse("prediction.html", {"request":request,"probalities":probalities, "predicted": predicted,"probality":probality})


#%%