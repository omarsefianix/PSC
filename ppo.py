# -*- coding: utf-8 -*-
"""ppo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PbZMJb5iFTmPCgCSJaas33oqqtp6QzwW
"""
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

##ajouter le nom de l'environnement à la place de 'envperso'
gym.register(
    id='Envperso',
    entry_point='__main__:flotte_gym',
)
##s'il n'est pas enregistré

# Création de l'environnement avec des environnements parallèles
vec_env = gym.make("FlotteBallons", max_episode_steps=1000, n=n_agents, show=False, req=req)
vec_env = DummyVecEnv([lambda: fg])

# Initialisation du modèle PPO avec une politique MLP
model = PPO("MlpPolicy", vec_env, verbose=1)

# Apprentissage du modèle sur 25000 étapes
model.learn(total_timesteps=25000)

# Sauvegarde du modèle
model.save("ppo_env")

# Chargement du modèle sauvegardé
model = PPO.load("ppo_env")

# Réinitialisation de l'environnement
obs = vec_env.reset()

# Boucle principale pour prédire et rendre les actions
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    ##vec_env.render("human") (pas sûr de si ca convient)
    ##on peut ajouter une condition pour finir la boucle

    # Si l'un des environnements est terminé, réinitialiser l'environnement
    if dones.any():
        obs = vec_env.reset()
