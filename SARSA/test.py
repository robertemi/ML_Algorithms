from Agent import Agent
from Environment import Environment

def train_agent():
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
    agent.sarsa(100)
    return agent

agent = train_agent()
best_path = agent.get_best_path(100)
print(best_path)
agent.path_visualization(best_path)
