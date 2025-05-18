from SARSA.Environment import Environment
import numpy as np
import random

class Agent:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = {}

    def q(self, state, action):
        # Return 0.0 if (state, action) isn't in Q-table yet
        return self.q_table.get((state, action), 0.0)

    '''
    epsilon greedy policy wrt q
    epsilon between (0, 1)
    '''
    def policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.choice(self.actions)
        else:
            #get q values for current state
            q_values = [self.q(state, a) for a in self.actions]

            #get greedy action for current state
            return self.actions[np.argmax(q_values)]

    def sarsa(self, episodes):
        '''
        Choose a random epsilon from (0, 1)
        Choose a random alpha (learning rate) from (0, 1)
        Choose a random gamma (discount factor) from (0, 1)
        '''
        epsilon = 0.1
        alpha = 0.1
        gamma = 0.9
        for _ in range(episodes):
            # init start state
            current_state = (self.environment.start_x, self.environment.start_y)
            action = self.policy(current_state, epsilon)

            # define max steps per episode to prevent infinite loop
            max_steps = 100
            while current_state != (self.environment.goal_x, self.environment.goal_y) and max_steps > 0:
                max_steps -= 1
                next_state, reward = self.environment.step(current_state, action)
                next_action = self.policy(next_state, epsilon)

                self.q_table[current_state, action] = self.q(current_state, action) + alpha * (reward + gamma * self.q(next_state, next_action) - self.q(current_state, action))
                current_state = next_state
                action = next_action




    