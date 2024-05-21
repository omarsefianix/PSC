import environnement.ballon as ballon
import environnement.parametres_ballon as pb
import environnement.parametres_air as pa
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import copy
from environnement.air import Air

class Flotte:
    def __init__(self, n:int, air:Air, time:dict, target:list, show = False, d_min=50) -> None:
        self.n = n
        self.list_ballon = []
        self.trajectory = [[] for i in range(self.n)]
        self.show = show
        self.time = time
        self.d_min=d_min
        for _ in range(n):
            lat = pa.update_longitude(target[0] + 180000 * rd.gauss(0, self.d_min)/(pb.R * np.pi))
            lon = (target[1] + 180000 * rd.gauss(0, self.d_min)/(np.cos(lat * np.pi/180) * pb.R * np.pi))%360
            z = pb.conversion_z_to_p(rd.uniform(15000, 20000))
            self.list_ballon.append(ballon.Ballon(air, time, [lat, lon], z, target, d_min=self.d_min))
            if(self.show):
                self.trajectory[_].append([lat, lon, pb.conversion_p_to_z(z)])
        self.target = target

    def reinitialise(self, time:dict, target:list, air:Air):
        self.trajectory = [[] for i  in range(self.n)]
        self.time = time
        self.target=target
        for _ in range(self.n):
            lat = pa.update_longitude(target[0] + 180000 * rd.gauss(0, self.d_min)/(pb.R * np.pi))
            lat = max(min(lat, self.air.latitude[-1]), self.air.latitude[0])
            lon = (target[1] + 180000 * rd.gauss(0, self.d_min)/(np.cos(lat * np.pi/180) * pb.R * np.pi))%360
            lon = max(min(lon, self.air.longitude[-1]), self.air.longitude[0])
            z = pb.conversion_z_to_p(rd.uniform(15000, 20000))
            self.list_ballon[_].reinitialise([lat,lon],z,self.time,self.target,air)
            if(self.show):
                self.trajectory[_].append([lat, lon, pb.conversion_p_to_z(z)])

    def copy(self):
        copie = Flotte(self.n, self.list_ballon[0].air.data_vent, copy.deepcopy(self.time), self.target, show = self.show)
        copie.list_ballon = []
        for ind in range(self.n):

            copie.list_ballon.append(self.list_ballon[ind].copy())
        copie.trajectory = copy.deepcopy(self.trajectory)
        return copie

    def get_reward(self):
        ans = [self.list_ballon[k].get_reward() for k in range(self.n)]
        return np.max(ans)

    def next_state(self, actions):
        self.time['steps'] += pb.dt
        if(self.time['steps']%self.air.timestep == 0):
            pb.update_time(self.time)
        for k in range(self.n):
            self.list_ballon[k].next_state(actions[k])
            if(self.show):
                self.trajectory[k].append([self.list_ballon[k].pos[0], self.list_ballon[k].pos[1], pb.conversion_p_to_z(self.list_ballon[k].z)])
        return self.get_reward()
    
    def get_inputs(self):
        altitudes = []
        charges = []
        distances = []
        bearings = []
        lights = []
        actions = []
        for k in range(self.n):
            l = self.list_ballon[k].get_inputs()
            actions.append(l.pop())
            lights.append(l.pop())
            bearings.append(l.pop())
            distances.append(l.pop())
            charges.append(l.pop())
            altitudes.append(l.pop())
        return [altitudes, charges, distances, bearings, lights, actions]
    
    def set_show(self, show):
        self.show = show

    def plot(self, title = ''):
        if(self.show):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for k in range(self.n):
                l = self.trajectory[k]
                lat = [l[i][1] for i in range(len(l))]
                lon = [l[i][0] for i in range(len(l))]
                z = [l[i][2] for i in range(len(l))]
                ax1.scatter(lat[0], lon[0], color = 'blue', label='Position initiale')
                # ax1.scatter([lat[i] for i in range(0,len(l), 500)], [lon[i] for i in range(0,len(l), 500)])
                ax1.plot(lat, lon)
                ax1.scatter(lat[-1], lon[-1], color = 'orange', label='Position finale')
                ax2.plot(z)
            ax1.set_title('Trajectoire')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            dl=2.5
            ax1.set_xlim(self.target[1] -dl , self.target[1] +dl) 
            ax1.set_ylim(self.target[0] -dl , self.target[0] +dl) 
            ax1.scatter(self.target[1], self.target[0], color='red', label='Target')
            circle = plt.Circle([self.target[1], self.target[0]], self.d_min*1000 * 180/(pb.R * np.pi), color='green', fill=False, label='Objective')
            ax1.add_patch(circle)
            ax1.set_aspect('equal')

            ax2.set_title('Altitude')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Altitude')
            plt.tight_layout()
        else:
            lat = [self.list_ballon[i].pos[0] for i in range(self.n)]
            lon = [self.list_ballon[i].pos[1] for i in range(self.n)]
            plt.scatter(lon, lat, color='blue', label='Balloons')

            # Plot the red point at the center
            plt.scatter(self.target[1], self.target[0], color='red', label='Target')

            # Plot the circle around the center
            circle = plt.Circle([self.target[1], self.target[0]], self.d_min*1000 * 180/(pb.R * np.pi), color='green', fill=False, label='Objective')
            plt.gca().add_patch(circle)

            # Set aspect ratio to equal to get a circular plot
            plt.axis('equal')

            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Add legend
            plt.legend()

        # Show plot
        plt.title(title)
        plt.show()