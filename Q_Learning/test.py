from Agent import Agent
from Environment import Environment

def train_agent1():
    env = Environment(9, 9, 0, 0, 8, 3)
    env.obstacles.add((1, 3))
    env.obstacles.add((2, 3))
    env.obstacles.add((2, 4))
    env.obstacles.add((4, 2))
    env.obstacles.add((3, 6))
    env.obstacles.add((3, 7))
    env.obstacles.add((5, 8))
    env.obstacles.add((6, 3))
    env.obstacles.add((7, 1))
    env.obstacles.add((7, 2))
    env.obstacles.add((7, 6))
    env.obstacles.add((7, 7))
    env.obstacles.add((8, 5))

    agent = Agent(env)
    agent.qLearn(100)
    return agent


def train_agent2():
    env = Environment(9, 9, 0, 0, 8, 3)
    env.obstacles.add((1, 3))
    env.obstacles.add((2, 3))
    env.obstacles.add((2, 4))
    env.obstacles.add((4, 2))
    env.obstacles.add((3, 6))
    env.obstacles.add((3, 7))
    env.obstacles.add((5, 8))
    env.obstacles.add((6, 3))
    env.obstacles.add((7, 1))
    env.obstacles.add((7, 2))
    env.obstacles.add((7, 6))
    env.obstacles.add((7, 7))
    env.obstacles.add((8, 5))

    agent = Agent(env)
    agent.qLearn2(100, 0.1)
    return agent

agent = train_agent1()
best_path = agent.get_best_path(100)
print(f'Epsilon greedy policy')
print(best_path)
agent.path_visualization(best_path)


agent2 = train_agent2()
best_path2 = agent.get_best_path(100)
print(f'Boltzmann policy')
print(best_path2)
agent.path_visualization(best_path2)