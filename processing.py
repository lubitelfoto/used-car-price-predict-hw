import pandas as pd
import numpy as np
import random
import re

random.seed(42)
np.random.seed(42)


mileage_mean = 27.02
engine_mean = 1248.0
max_power_mean = 81.86
torque_mean = 153.0
rpm_mean = 3100.0
seat_mean = 5.0


def processing(dfc):
    df = dfc.copy()
    mileage = list()
    for value in df.mileage:
        if type(value)==str:
            if value.endswith("km/kg"):
                mileage.append(float(value.replace("km/kg", "")))
            if value.endswith("kmpl"):
                mileage.append(float(value.replace("kmpl", ""))*1.4)
        else:
            mileage.append(value)
    df.mileage = mileage
    df.mileage.fillna(mileage_mean, inplace=True)
    df.mileage = df.mileage.astype(float)
    df.engine = df.engine.apply(lambda x: x.replace("CC", "") if type(x)==str else x)
    df.engine.fillna(engine_mean, inplace=True)
    df.engine = df.engine.astype(int)
    
    df.max_power = df.max_power.apply(lambda x: x.replace("bhp", "") if type(x)==str else x)
    df.max_power = df.max_power.replace(r'^\s+$', np.nan, regex=True)
    df.max_power.fillna(max_power_mean, inplace=True)
    df.max_power = df.max_power.astype(float)
    
    torq = list()
    rpm = list()
    for val in df.torque:
        if type(val)==str:
            val = val.lower()
            val = val.replace(",",".")
            val = val.replace("+/-500rpm", "")
            if val.endswith("(kgm@ rpm)"):
                val = val.replace(" ", "")
                torque = val[:-9].split("@")
                torq.append(float(torque[0]))
                rpm.append(torque[1])
            else:
                val = val.replace(" ", "")
                torque = re.split("@|at|/", val)
                record = [np.NAN,np.NAN]
                for item in torque:
                    if item.endswith("nm"):
                        record[0] = float(re.findall("\d{1,}", item)[0])
                    if item.endswith("kgm"):
                        record[0] = float(item.replace("kgm", ""))*9.8067
                    if item.endswith("rpm"):
                        record[1] = item.replace("rpm","")
                torq.append(record[0])        
                rpm.append(record[1])
        else:
            torq.append(val)
            rpm.append(np.NAN)
    df.torque = torq
    df.torque.fillna(torque_mean, inplace=True)
    df.torque = df.torque.astype(float)
    for i, val in enumerate(rpm):
        val = str(val)
        val = val.replace(".","")
        values = re.findall("\d{1,}", val)
        if len(values)==2:
            rpm[i]=int(values[0])+int(values[1])/2
        else: 
            numbers = re.findall("\d{1,}", val)
            if numbers:
                rpm[i] = numbers[0]
            else:
                rpm[i] = np.NAN
    df['max_torque_rpm'] = rpm
    df.max_torque_rpm.fillna(rpm_mean, inplace=True)
    df.max_torque_rpm = df.max_torque_rpm.astype(float)

    df.seats.fillna(seat_mean, inplace=True)
    df.seats = df.seats.astype(int)
    
    return df


def hp_per_liter(df):
    return df.max_power/(df.engine/1000)


def power_year(df_orig):
    df = df_orig.copy()
    np.power(df_orig.year,2)
    return df


def add_lux_brands(df):
    luxury_brands = ["audi", "aston", "bentley", "bmw", "ferrari", "jaguar",
                     "lamborghini", "land", "maserati", "mercedes-benz", "porsche",
                     "rolls-royce", "lexus", "genesis", "cadillac"]
    brands = df.name.map(lambda s: re.match('^\S+\s', s.lower()).group())
    brands = brands.map(lambda s: s.replace(" ",""))
    lux_brand_col = list()
    for brand in brands:
        lux_brand_col.append(brand in luxury_brands)
    return lux_brand_col

def del_columns(df_orig):
    df = df_orig.copy()
    del_columns = ['transmission_Manual', 'fuel_CNG',
                   'seller_type_Trustmark Dealer', 'owner_Test Drive Car']
    return df.drop(columns=del_columns)


def scale_df(df_orig, scaler):
    df = df_orig.copy()
    col_to_scale = ['year', 'km_driven',
                    'mileage', 'engine', 'max_power', 'torque',
                    'max_torque_rpm']
    df_scaled = scaler.transform(df.loc[:, col_to_scale].to_numpy())
    df.loc[:, col_to_scale] = df_scaled
    return df
