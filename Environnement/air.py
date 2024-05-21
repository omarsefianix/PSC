import numpy as np
import environnement.parametres_air as pa


class Air:
    def __init__(self, vent, request, db='ERA5'):
        self.db = db
        lg0, lt0 = request["coords"]["longitude"], request["coords"]["latitude"]
        nlg, nlt = request["deltas"]["longitude"]["nb"], request["deltas"]["latitude"]["nb"]
        dlg, dlt = request["deltas"]["longitude"]["dl"], request["deltas"]["latitude"]["dl"]
        if self.db == "ERA5":
            self.pressure = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150., 175., 200., 225., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 775., 800., 825., 850., 875., 900., 925., 950., 975., 1000.])
            self.timestep=1
        else:
            self.pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
            self.timestep=6

        self.longitude = np.array([lg0 + i * dlg for i in range(nlg)])      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = np.array([lt0 + i * dlt for i in range(nlt)])  #de -90 à 90 avec un pas de 2.5
        self.data_vent = vent #vent[time][pressure][longitude][latitude]->[u, v]
        #base de donnée: mesure toute les 6h
        #Pression en hPa

    def get_vent(self, pos: list, time: dict, target) -> list:
        t = int(time['steps']//self.timestep)
        low_lon = pa.recherche(self.longitude, pos[1])
        low_lat = pa.recherche(self.latitude, pos[0])
        request_vent = self.data_vent[t : t + 2, :, low_lon : low_lon + 2, low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': self.pressure, 'latitude': [self.latitude[low_lat], self.latitude[low_lat + 1]], 'longitude': [self.longitude[low_lon], self.longitude[low_lon + 1]]}
        vent = pa.interpolation(pos, self.pressure, time['steps'], request_vent, request_bounds)
        angle = np.angle(complex(target[0] - pos[0], target[1] - pos[1]))
        ans = []
        for k in range(len(vent)):
            x = np.linalg.norm(vent[k])
            ans.append([x/(x + 30), np.abs(((np.angle(complex(vent[k][1], vent[k][0])) - angle + np.pi)%(2*np.pi) - np.pi)/np.pi)])
        return np.array(ans)
    
    def new_pos(self, pos: list, pressure: float, time: dict, dt: float, target: tuple):
        t = int(time['steps']//self.timestep)
        low_lon = pa.recherche(self.longitude, pos[1])
        low_lat = pa.recherche(self.latitude, pos[0])
        low_p = pa.recherche(self.pressure, pressure)
        request_vent = self.data_vent[t : t + 2, low_p : low_p + 2, low_lon : low_lon + 2, low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': [self.pressure[low_p], self.pressure[low_p + 1]], 'latitude': [self.latitude[low_lat], self.latitude[low_lat + 1]], 'longitude': [self.longitude[low_lon], self.longitude[low_lon + 1]]}
        vent = pa.interpolation(pos, [pressure], time['steps'], request_vent, request_bounds)[0]
        vent[0], vent[1] = vent[1], vent[0]
        pos[0] += 180 * vent[0] * dt * 3600/(np.pi * pa.R)
        pos[0] = pa.update_longitude(pos[0])
        pos[0] = max(min(pos[0], self.latitude[-1]), self.latitude[0])

        pos[1] += 180 * vent[1] * dt * 3600/(np.pi * pa.R)
        pos[1] = pos[1]%360
        pos[1] = max(min(pos[1], self.longitude[-1]), self.longitude[0])
        vector = np.array(target) - np.array(pos)
        return np.array([np.dot(vent, vector), np.cross(vent, vector)])/(np.linalg.norm(vent) * np.linalg.norm(vector))
    
    def update_vent(self, vent, request):
        lg0, lt0 = request["coords"]["longitude"], request["coords"]["latitude"]
        nlg, nlt = request["deltas"]["longitude"]["nb"], request["deltas"]["latitude"]["nb"]
        dlg, dlt = request["deltas"]["longitude"]["dl"], request["deltas"]["latitude"]["dl"]
        if self.db == "ERA5":
            self.pressure = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150., 175., 200., 225., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 775., 800., 825., 850., 875., 900., 925., 950., 975., 1000.])
            self.timestep=1
        else:
            self.pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
            self.timestep=6

        self.longitude = np.array([lg0 + i * dlg for i in range(nlg)])      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = np.array([lt0 + i * dlt for i in range(nlt)])  #de -90 à 90 avec un pas de 2.5
        self.data_vent = vent

    def adjust_pression(self, p):
        return max(min(p, self.pressure[-1]), self.pressure[0])