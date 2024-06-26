import numpy as np
import datetime
import environnement.parametres_ballon as pb
from scipy.interpolate import RegularGridInterpolator
R = pb.R
DELTAS={
    "time": {
        "day":9,
        "hour":23,
    },
    "latitude":{
        "nb":20,
        "dl": .25,
    },
    "longitude":{
        "nb":20,
        "dl":.25,
    }
}

##Récupération du vent

def interpolation(pos, pressure, hour, vent, request_bounds, timestep=1) -> list:#vent liste des liste de vents a interpoler
    interp = RegularGridInterpolator((request_bounds["time"], request_bounds["pressure"], request_bounds["longitude"], request_bounds["latitude"]), vent)
    pressure = np.array([pressure]).flatten()
    coords = np.array([[hour, p, pos[1], pos[0]] for p in pressure])
    ret = interp(coords)
    return ret
    ans = []
    test = False
    longueur = 1
    if(len(pressure) == 1):
        test = True
        longueur = 2
    for l in range(len(pressure)):
        sum = np.zeros(2)
        for t in range(len(request_bounds['time'])):
            pt = 1 - np.abs(hour%timestep - t * timestep)/timestep
            for p in range(longueur):
                pp = 1 - test * np.abs(request_bounds['pressure'][p] - pressure[0])/(request_bounds['pressure'][1] - request_bounds['pressure'][0])
                for lon in range(len(request_bounds['longitude'])):
                    plon = 1 - np.abs(request_bounds['longitude'][lon] - pos[1])/2.5
                    for lat in range(len(request_bounds['latitude'])):
                        plat = 1 - np.abs(request_bounds['latitude'][lat] - pos[0])/2.5
                        try :
                            sum += vent[t][l + p][lon][lat] * pt * pp * plon * plat
                        except:
                            print(t, l+p, lon, lat, vent.shape)
                            raise ValueError
        ans.append(sum)
    return ans

def recherche(liste: list, x: float) -> int:
    low = 0
    high = len(liste) - 1
    while(high - low > 1):
        m = liste[(high + low)//2]
        if(m > x):
            high = (high + low)//2
        else:
            low = (high + low)//2
    return low

def date_vent(time):
    start_date = datetime.datetime(year = time['year'], month = time['month'], day = time['day'], hour = time['hour'])
    copy = time.copy()
    pb.update_time(copy)
    end_date = datetime.datetime(year = copy['year'], month = copy['month'], day = copy['day'], hour = copy['hour'])
    return [start_date, end_date]

def update_longitude(teta):
    teta = ((teta + 90) % 360) - 90
    if teta > 90:
        teta = 180 - teta
    elif teta < -90:
        teta = -180 -teta
    return teta
