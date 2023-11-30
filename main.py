import json
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from typing import List
import uvicorn
from processing import processing, hp_per_liter, power_year, add_lux_brands, del_columns, scale_df

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
#   кажется странным что колонка цены есть в сервисе который ее предсказывает
#   selling_price: int
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


json_car = """{"name" : "Hyundai Grand i10 Magna", "year" : 2015, "km_driven" : 70000, "fuel" : "Petrol", "seller_type" 
            : "Individual", "transmission" : "Manual", "owner" : "First Owner", "mileage" : "18.9 kmpl", "engine" : 
            "1197 CC", "max_power" : "82 bhp", "torque" : "114Nm@ 4000rpm", "seats" : 5.0}"""

item = Item.model_validate_json(json_car)


model = ElasticNet()
params = pickle.load(open("model.pickle", "rb"))
model.coef_ = params["coef"]
model.set_params(**params["params"])
model.feature_names_in_ = params["feature"]
model.intercept_ = params["intercept"]

scaler = pickle.load(open("scaler.pickle", "rb"))


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    dict_item = item.model_dump()
    df = pd.DataFrame(dict_item, columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
                                     'seats', 'max_torque_rpm', 'fuel', 'seller_type', 'transmission', 'owner',]
                      , index=[0])
    name_df = df.copy()
    df = df.drop(columns='name')
    df = processing(df)
    df['hp_per_liter'] = hp_per_liter(df)
    df = power_year(df)
    df = pd.get_dummies(df)
    dummy_col = {'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Dealer', 'seller_type_Individual',
                 'seller_type_Trustmark Dealer', 'transmission_Automatic', 'transmission_Manual', 'owner_First Owner',
                 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner', }
    col = list(dummy_col - set(df.columns))
    df[col] = 0
    df = del_columns(df)
    df = scale_df(df, scaler)
    col = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
           'seats', 'max_torque_rpm', 'fuel_Diesel', 'fuel_LPG',
           'fuel_Petrol', 'seller_type_Dealer', 'seller_type_Individual', 'transmission_Automatic', 'owner_First Owner',
           'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Third Owner',
           'hp_per_liter']
    df = df[col]
    df["is_luxury_brand"] = add_lux_brands(name_df)
    predicts = model.predict(df)
    return predicts


@app.post("/predict_items")
def predict_items(file: UploadFile = File()) -> List[float]:
    df = pd.read_csv(file.file)
    name_df = df.copy()
    df = df.drop(columns='name')
    df = processing(df)
    df['hp_per_liter'] = hp_per_liter(df)
    df = power_year(df)
    df = pd.get_dummies(df)
    dummy_col = {'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Dealer', 'seller_type_Individual',
                 'seller_type_Trustmark Dealer', 'transmission_Automatic', 'transmission_Manual', 'owner_First Owner',
                 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner',}
    col = list(dummy_col - set(df.columns))
    df[col] = 0
    df = del_columns(df)
    df = scale_df(df, scaler)
    col = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
           'seats', 'max_torque_rpm', 'fuel_Diesel', 'fuel_LPG',
           'fuel_Petrol', 'seller_type_Dealer', 'seller_type_Individual', 'transmission_Automatic', 'owner_First Owner',
           'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Third Owner',
           'hp_per_liter']
    df = df[col]
    df["is_luxury_brand"] = add_lux_brands(name_df)
    predicts = model.predict(df)
    return predicts


uvicorn.run(app)
