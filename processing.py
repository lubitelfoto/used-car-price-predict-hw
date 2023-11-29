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


def processing(dfc, train_df=None):
    df = dfc.copy()
    
    mileage = list()
    for value in df.mileage:
        if type(value)==str:
            if value.endswith("km/kg"):
                mileage.append(float(value.replace("km/kg","")))
            if value.endswith("kmpl"):
                mileage.append(float(value.replace("kmpl", ""))*1.4)
        else:
            mileage.append(value)
    df.mileage = mileage
    if train_df is not None:
        mileage_mean = train_df[train_df.mileage.notna()].mileage.astype(float).median()
    else:
        mileage_mean = df[df.mileage.notna()].mileage.astype(float).median()
    df.mileage.fillna(mileage_mean, inplace=True)
    df.mileage = df.mileage.astype(float)
    
    df.engine = df.engine.apply(lambda x: x.replace("CC", "") if type(x)==str else x) 
    if train_df is not None:
        engine_mean = train_df[train_df.engine.notna()].engine.astype(float).median()
    else:
        engine_mean = df[df.engine.notna()].engine.astype(float).median()
    df.engine.fillna(engine_mean, inplace=True)
    df.engine = df.engine.astype(int)
    
    df.max_power = df.max_power.apply(lambda x: x.replace("bhp", "") if type(x)==str else x)
    df.max_power = df.max_power.replace(r'^\s+$', np.nan, regex=True)
    if train_df is not None:
        max_power_mean = train_df[train_df.max_power.notna()].max_power.astype(float).median()
    else:
        max_power_mean = df[df.max_power.notna()].max_power.astype(float).median()
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
    if train_df is not None:
        torque_mean = train_df[train_df.torque.notna()].torque.median()
    else:
        torque_mean = df[df.torque.notna()].torque.median()
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
    if train_df is not None:
        rpm_mean = train_df[train_df.max_torque_rpm.notna()].max_torque_rpm.astype(int).median()
    else:
        rpm_mean = df[df.max_torque_rpm.notna()].max_torque_rpm.astype(int).median()
    df.max_torque_rpm.fillna(rpm_mean, inplace=True)
    df.max_torque_rpm = df.max_torque_rpm.astype(float)
    
    if train_df is not None:
        seat_mean = train_df[train_df.seats.notna()].seats.astype(int).median()
    else:
        seat_mean = df[df.seats.notna()].seats.astype(int).median()
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