import environnement.ballon as ballon
import environnement.parametres_ballon as pb
import environnement.parametres_air as pa
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import copy
import gymnasium as gym
import environnement.flotte as flotte
from environnement.wind_storage import WindStorage
from environnement.air import Air

from gymnasium import spaces

class Flotte_Gym(gym.Env, flotte.Flotte):
    def __init__(self, n: int, req:dict=None, target:list=None, show=False, dt:int = 2, deltas:dict=pa.DELTAS, d_min=50, *args, **kwargs) -> None:

        self.wind_storage = WindStorage()
        if target == None:
            self.vent, self.req = self.wind_storage.get_random_wind(deltas) if req==None else self.wind_storage.get_wind(req), req
            self.target = [self.req["coords"]["latitude"] + self.req["deltas"]["latitude"]["nb"]*self.req["deltas"]["latitude"]["dl"]/2, self.req["coords"]["longitude"] + self.req["deltas"]["longitude"]["nb"]*self.req["deltas"]["longitude"]["dl"]/2]
        else :
            self.target = target
            coords={
                "latitude": target[0] - deltas["latitude"]["nb"]*deltas["latitude"]["dl"]/2,
                "longitude": target[1] - deltas["longitude"]["nb"]*deltas["longitude"]["dl"]/2
            }
            start_date=req["start_date"] if "start_date" in req else None
            self.vent, self.req = self.wind_storage.get_random_wind(deltas, start_date=start_date, coords=coords)
        
        self.time = self.req['start_date']
        self.time["steps"] = 0
        self.air = Air(self.vent, self.req)

        super().__init__(n, self.air, self.time, self.target, show, d_min=d_min, *args, **kwargs)
        self.dt = dt if dt >= 1 else 1
        self.action_space = spaces.Box(-1, 1, (n,))
        wind_shape = list(np.moveaxis(self.vent, -1, -4).shape)
        wind_shape[0] = self.dt
        self.observation_space = spaces.Dict({
            "vent":spaces.Box(low=-np.inf, high=np.inf, shape=wind_shape), 
            "obj": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.target),)), 
            "ballons_altitudes":spaces.Box(low=0, high=1, shape=(self.n,)),
            "ballons_charges":spaces.Box(low=0, high=1, shape=(self.n,)),
            "ballons_distances":spaces.Box(low=0, high=np.inf, shape=(self.n,)),
            "ballons_bearings":spaces.Box(low=-np.inf, high=np.inf, shape=(n,2)),
            "ballons_lights":spaces.MultiBinary(n=n),
            "ballons_actions":spaces.Box(low=-1, high=1, shape=(self.n,))
        })

    def state(self):
        vent_act = self.vent[int(self.time["steps"]/self.air.timestep):int(self.time["steps"]/self.air.timestep)+self.dt]
        altitudes, charges, distances, bearings, lights, actions = self.get_inputs()
        return {
            "vent": np.moveaxis(vent_act, -1, -4),
            "obj": self.target, 
            "ballons_altitudes":altitudes, 
            "ballons_charges":charges, 
            "ballons_distances":distances, 
            "ballons_bearings":bearings, 
            "ballons_lights":lights, 
            "ballons_actions":actions
        }
    
    def step(self, action):
        reward = self.next_state(action)
        truncated = (self.time["steps"] + self.dt) >= len(self.vent)
        return self.state(), reward, False, truncated, {}
    
    def reset(self, seed:int=None, req:dict=None, target:list=None, deltas:dict=pa.DELTAS):
        super().reset(seed=seed)
        if self.time["steps"] == 0:
            req = self.req
        if target == None:
            self.vent, self.req = self.wind_storage.get_random_wind(deltas) if req==None else (self.wind_storage.get_wind(req), req)
            self.target = [self.req["coords"]["latitude"] + self.req["deltas"]["latitude"]["nb"]*self.req["deltas"]["latitude"]["dl"]/2, self.req["coords"]["longitude"] + self.req["deltas"]["longitude"]["nb"]*self.req["deltas"]["longitude"]["dl"]/2]
        else :
            self.target = target
            coords={
                "latitude": target[0] - deltas["latitude"]["nb"]*deltas["latitude"]["dl"]/2,
                "longitude": target[1] - deltas["longitude"]["nb"]*deltas["longitude"]["dl"]/2
            }
            start_date=req["start_date"] if "start_date" in req else None
            self.vent, self.req = self.wind_storage.get_random_wind(deltas, start_date=start_date, coords=coords)
        self.time["steps"]=0
        self.air.update_vent(self.vent, self.req)
        super().reinitialise(self.time, self.target, self.air)
        return self.state(), {}
