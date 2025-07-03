from SARSA.Environment import Environment
import numpy as np
import random
import matplotlib.pyplot as plt

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
        all_paths = []
        for _ in range(episodes):
            # init start state
            current_state = (self.environment.start_x, self.environment.start_y)
            action = self.policy(current_state, epsilon)

            # define max steps per episode to prevent infinite loop
            max_steps = 100
            path_per_episode = [current_state]
            while current_state != (self.environment.goal_x, self.environment.goal_y) and max_steps > 0:
                max_steps -= 1
                next_state, reward = self.environment.step(current_state, action)
                next_action = self.policy(next_state, epsilon)

                self.q_table[current_state, action] = self.q(current_state, action) + alpha * (reward + gamma * self.q(next_state, next_action) - self.q(current_state, action))
                current_state = next_state
                action = next_action
                path_per_episode.append(current_state)
            all_paths.append(path_per_episode)


    def get_best_path(self, max_steps=100):
        current_state = (self.environment.start_x, self.environment.start_y)
        best_path = [current_state]
        visited = set()

        while current_state != (self.environment.goal_x, self.environment.goal_y):
            if max_steps == 0:
                break
            
            visited.add(current_state)

            # greedy action selection 
            q_values = [self.q(current_state, a) for a in self.actions]
            greedy_action = self.actions[np.argmax(q_values)]
            next_state, reward = self.environment.step(current_state, greedy_action)

            # agent stuck in a loop
            if next_state in visited:
                break

            best_path.append(next_state)
            current_state = next_state
            max_steps -= 1
            
        return best_path
    

    def path_visualization(self, best_path, ax=None):
        width, height = self.environment.width, self.environment.height

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
            show_plot = True

        # Draw grid cells
        for x in range(width):
            for y in range(height):
                color = 'black' if (x, y) in self.environment.obstacles else 'white'
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray'))

        # Draw best path (in green)
        for i in range(len(best_path) - 1):
            x1, y1 = best_path[i]
            x2, y2 = best_path[i + 1]
            ax.arrow(x1 + 0.5, y1 + 0.5, x2 - x1, y2 - y1,
                    head_width=0.2, length_includes_head=True, color='green')

        # Mark start and goal
        sx, sy = self.environment.start_x, self.environment.start_y
        gx, gy = self.environment.goal_x, self.environment.goal_y
        ax.text(sx + 0.3, sy + 0.3, 'S', fontsize=14, color='red')
        ax.text(gx + 0.3, gy + 0.3, 'F', fontsize=14, color='darkgreen')

        # Formatting
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, width + 1, 1))
        ax.set_yticks(np.arange(0, height + 1, 1))
        ax.grid(True)
        ax.invert_yaxis()  # Use ax, not plt.gca()
        # Do not set title here if you want to set it outside

        if show_plot:
            plt.show()




    