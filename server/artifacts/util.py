import pickle
import joblib
import json
import numpy as np
import pandas as pd
__brand=None
__gearbox=None
__origin=None
__type=None

def get_predict_car_price(manufacture_date, seats, mileage_v2, color, model, brand, origin,type,gearbox):
    try:
        loc_index=__brand.index(brand.lower())
    except:
        loc_index=-1

    try:
        loc_index1=__gearbox.index(gearbox.lower())
    except:
        loc_index=-1

    try:
        loc_index=__origin.index(origin.lower())
    except:
        loc_index=-1

    try:
        loc_index=__type.index(type.lower())
    except:
        loc_index=-1

    if loc_index==0:
        brand=0
    elif loc_index==1:
        brand=1
    elif loc_index==2:
        brand=2
    elif loc_index==3:
        brand=3
    elif loc_index==4:
        brand=4
    elif loc_index==5:
        brand=5
    elif loc_index==6:
        brand=6
    elif loc_index==7:
        brand=7
    elif loc_index==8:
        brand=8
    elif loc_index==9:
        brand=9
    elif loc_index==10:
        brand=10
    elif loc_index==11:
        brand=11

    if loc_index1==0:
        gearbox=0
    elif loc_index1==1:
        gearbox=1
    elif loc_index1==2:
        gearbox=2
    elif loc_index1==3:
        gearbox=3

    if loc_index2==0:
        origin=0
    elif loc_index2==1:
        origin=1
    elif loc_index2==2:
        origin=2
    elif loc_index2==3:
        origin=3
    elif loc_index2==4:
        origin=4
    elif loc_index2==5:
        origin=5
    elif loc_index2==6:
        origin=6
    elif loc_index2==7:
        origin=7
    elif loc_index2==8:
        origin=8
    elif loc_index2==9:
        origin=9

    if loc_index3==0:
        type=0
    elif loc_index3==1:
        type=1
    elif loc_index3==2:
        type=2
    elif loc_index3==3:
        type=3
    elif loc_index3==4:
        type=4
    elif loc_index3==5:
        type=5
    elif loc_index3==6:
        type=6
    elif loc_index3==7:
        type=7
    elif loc_index3==8:
        type=8
'''
new_sample=[manufacture_date, seats, mileage_v2, color, model, brand, origin,type,gearbox]
x=pd.DataFrame(new_sample)
x.columns= __columns
'''
predict=str(__model.car_price.predict(x)[0])
def load_saved_artifacts():
    print('Loading saved artifacts... start')
    global __brand
    global __gearbox
    global __origin
    global __type
    with open('./artifacts/brand_value.json', 'r') as f:
        __brand=json.load(f)['brand_value']
    with open('./artifacts/gearbox_value.json', 'r') as f:
        __gearbox=json.load(f)['gearbox_value']
    with open('./origin/brand_value.json', 'r') as f:
        __origin=json.load(f)['origin_value']
    with open('./artifacts/type_value.json', 'r') as f:
        __type = json.load(f)['type_value']

    global __model
    if __model is None:
        with open('./artifacts/model_HR.sav', 'rb') as f:
            __model=joblib.load(f)
    print('loading saved artifacts..done')
def get_brand_names():
    return __brand
def get_gearbox():
    return __gearbox
def get_origin():
    return __origin
def get_type():
    return __type
def get_columns():
    return columns
if __name__=='__main__':
    load_saved_artifacts()