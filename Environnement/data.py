import datetime
import json
import numpy as np
import random

from environnement.parametres_air import DATA_STORAGE_FILE, DELTAS
from environnement.data_access.request_build.requests_manager import fetch,get,delete,metadata
import parametres_entrainement as pe
import environnement.parametres_ballon as pb
import environnement.parametres_air as pa

jours = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

request_item = {
            'dataset': "ERA5",
            'memory_limit': 200*1000*1000,
            'bounds': {
                'latitude': [],
                'longitude': [],
                'pressure':[],
                'time': []
            },
            'subsampling':{
                'longitude':1,
                'latitude':1,
                'pressure':1,
                'hour':1,
                'month':1,
                'day':1
            }
        }


# UTILS

def convert_to_time(time, dt=None):
    t0 = datetime.datetime(year = time['year'], month = time['month'], day = time['day'], hour = time['hour'])
    if dt is not None:
        t1 = t0 + datetime.timedelta(days=dt["day"], hours=dt["hour"])
        return t0, t1
    return t0

def get_storage_data(filename):
    with open(filename, "r") as f:
        data_in_storage = [json.loads(l.strip('\n')) for l in f if len(l.strip('\n'))>0]

    dates_stored=[]
    for d in data_in_storage:
        time = d["start_date"]
        dt = d["deltas"]["time"]
        t0, t1 = convert_to_time(time, dt)
        dates_stored.append([t0, t1])
    
    return data_in_storage, dates_stored

def add_storage_data(filename, req):
    with open(filename, "a") as f:
        f.write('\n')
        f.write(json.dumps(req))
        

def is_available(t_s, t_e, dates):
    for d in dates:
        t0, t1 = d[0], d[1]
        if (t_s >= t0 and t_s <= t1) or (t_e >= t0 and t_e <= t1):
            return False
    return True

def get_request_item(start_date, coords, deltas=DELTAS, db="ERA5"):
    assert "longitude" in coords and "latitude" in coords, "Coords must contain 'longitude' and 'latitude'."
    assert coords["longitude"] < 360 and coords["longitude"] >= 0 and coords["latitude"] <= 90 and coords["latitude"] >= -90, "Coords are off-limit."
    
    start_time = datetime.datetime(year = start_date['year'], month = start_date['month'], day = start_date['day'], hour = start_date['hour'])
    end_time = start_time + datetime.timedelta(days=deltas["time"]["day"], hours=deltas["time"]["hour"])
    request_item['bounds']['time'] = np.array([start_time, end_time])
    request_item['bounds']['latitude'] = np.array([
        coords["latitude"] + i*deltas["latitude"]["dl"] if coords["latitude"] + i*deltas["latitude"]["dl"] <= 90
        else coords["latitude"] + i*deltas["latitude"]["dl"] - 180
        for i in range(deltas["latitude"]["nb"])
    ])
    request_item['bounds']['longitude'] = np.array([
        coords["longitude"] + i*deltas["longitude"]["dl"] if coords["longitude"] + i*deltas["longitude"]["dl"] < 360
        else coords["longitude"] + i*deltas["longitude"]["dl"] - 360
        for i in range(deltas["longitude"]["nb"])
    ])
    request_item['bounds']['pressure'] = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150., 175., 200., 225., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 775., 800., 825., 850., 875., 900., 925., 950., 975., 1000.]) if db == "ERA5" else np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
    return request_item

# DATA GETTERS

def back_time(time):
    time['hour'] -= 6
    if(time['hour'] < 0):
        time['hour'] = time['hour']%24
        time['day'] -= 1
        if(time['day'] < 1):
            time['month'] = (time['month'] - 2)%12 + 1
            limite = jours[time['month'] - 1]
            if(time['month'] == 2 and time['year']%4 == 0):
                limite += 1
            time['day'] = limite
            if(time['month'] == 12):
                time['year'] -= 1

def request_vent(request_item):
    fetch(request_item)
    wind_data = get(request_item, skip_check=True)
    return wind_data['data']

def get_data(start_date):
    start_time = datetime.datetime(year = start_date['year'], month = start_date['month'], day = start_date['day'], hour = start_date['hour'])
    end_time = start_date
    for _ in range((int(pe.duration) + 2) * 4):
        pb.update_time(end_time)
    #back_time(end_time)
    end_time = datetime.datetime(year = end_time['year'], month = end_time['month'], day = end_time['day'], hour = end_time['hour'])
    request_item['bounds']['time'] = np.array([start_time, end_time])
    request_item['bounds']['latitude'] = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
    request_item['bounds']['longitude'] = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
    request_item['bounds']['pressure'] = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
    return request_vent(request_item)

def get_data_coords(start_date, coords, deltas=DELTAS, db="ERA5"):
    return request_vent(get_request_item(start_date, coords, deltas))

def get_random_downloaded_wind(filename=DATA_STORAGE_FILE):
    with open(filename, "r") as f:
        data_in_storage = [json.loads(l) for l in f]
    req = random.choice(data_in_storage)
    return request_vent(get_request_item(req["start_date"], req["coords"], req["deltas"])), req

# DATA GENERATOR

def generate_data(deltas):
    data_in_storage, dates_stored = get_storage_data(DATA_STORAGE_FILE)
    rd_start_date = None
    valid = False
    while not valid:
        rd_start_date = {
            "year":np.random.randint(low=1990, high=2024),
            "month":np.random.randint(low=1, high=13),
            "day":np.random.randint(low=1, high=29),
            "hour":0,
        }
        t0, t1 = convert_to_time(rd_start_date, deltas["time"])
        valid = is_available(t0, t1, dates_stored)
    rd_coords = {
        "longitude": np.random.randint(deltas['longitude']['nb'], 1440-deltas['longitude']['nb'])*.25,
        "latitude": -90 + np.random.randint(deltas['latitude']['nb'], 721-deltas['latitude']['nb'])*.25,
    }
    req = {
        "start_date":rd_start_date,
        "coords": rd_coords,
        "deltas": deltas
    }
    fetch(get_request_item(rd_start_date, rd_coords, deltas))
    add_storage_data(DATA_STORAGE_FILE, req)
