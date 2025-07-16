import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from SARSA.Agent import Agent as SARSA
from Q_Learning.Agent import Agent as Q
from Q_Learning.Environment import Environment

def build_env():
    env = Environment(9, 9, 0, 0, 3, 8)
    env.obstacles.update({
        (1, 3), (2, 3), (2, 4), (4, 2), (3, 6), (3, 7),
        (5, 8), (6, 3), (7, 1), (7, 2), (7, 6), (7, 7), (8, 5)
    })
    return env

def train_agent_q():
    env = build_env()
    q_agent = Q(env)
    q_agent.qLearn(1000)  
    return q_agent

def train_agent_sarsa():
    env = build_env()
    sarsa_agent = SARSA(env)
    sarsa_agent.sarsa(1000)  
    return sarsa_agent

q = train_agent_q()
sarsa = train_agent_sarsa()
env = build_env()

q_best = q.get_best_path(100)
sarsa_best = sarsa.get_best_path(100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
q.path_visualization(q_best, ax=ax1)
ax1.set_title("Q-Learning Best Path")
sarsa.path_visualization(sarsa_best, ax=ax2)
ax2.set_title("SARSA Best Path")
plt.show()
