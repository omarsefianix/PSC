import numpy as np
import json
from environnement.data import get_data_coords

def check_request(request):
    assert "start_date" in request and "coords" in request and "deltas" in request, "Request must contain 'start_date', 'coords' and 'deltas'."
    
    assert "year" in request["start_date"] and "month" in request["start_date"] and "day" in request["start_date"] and "hour" in request["start_date"], "Missing start_date info."
    assert "longitude" in request["coords"] and "latitude" in request["coords"], "Missing coords info."
    assert "time" in request["deltas"] and "latitude" in request["deltas"] and "longitude" in request["deltas"], "Missing deltas info."
    
    assert "day" in request["deltas"]["time"] and "hour" in request["deltas"]["time"], "Missing deltas.time info."
    assert "nb" in request["deltas"]["latitude"] and "dl" in request["deltas"]["latitude"]
    assert "nb" in request["deltas"]["longitude"] and "dl" in request["deltas"]["longitude"]

def compare_time_requests(r_a, r_b):
    for t in ["day", "hour"]:
        if r_a[t] > r_b[t]:
            return 1
        if r_a[t] < r_b[t]:
            return -1
    return 0

class WindStorage:
    def __init__(self):
        self.winds={}
    
    def get_wind(self, request, start=0, end=-1):
        '''
        request={
            start_date:{},
            coords:{},
            deltas:{
                time:{}
                latitude:{}
                longitude:{}
            }
        }
        '''
        check_request(request)

        key = json.dumps({
            'start_date':request['start_date'],
            'coords':request['coords'],
            'deltas':{
                'latitude':request['deltas']['latitude'],
                'longitude':request['deltas']['longitude'],
            }
        })
        if key not in self.winds:
            self.winds[key] = get_data_coords(start_date=request['start_date'], coords=request['coords'], deltas=request['deltas'])

        return self.winds[key][start:end]
    
    def get_random_wind(self, deltas, start=0, end=-1, start_date=None, coords=None):
        rd_start_date = {
            "year":np.random.randint(low=1990, high=2024),
            "month":np.random.randint(low=1, high=13),
            "day":np.random.randint(low=1, high=29),
            "hour":0,
        }
        rd_coords = {
            "longitude": np.random.randint(deltas['longitude']['nb'], 1440-deltas['longitude']['nb'])*.25,
            "latitude": -90 + np.random.randint(deltas['latitude']['nb'], 721-deltas['latitude']['nb'])*.25,
        }
        req = {
            "start_date":rd_start_date if start_date == None else start_date,
            "coords": rd_coords if coords == None else coords,
            "deltas": deltas
        }
        return self.get_wind(req, start, end), req