import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = {}

    def q(self, state, action):
        # Return 0.0 if (state, action) isn't in Q-table yet
        return self.q_table.get((state, action), 0.0)
    
    # needs two policies, one for exploration and one for exploitation

    # epsilon greedy policy for exploration
    # epsilon from the interval [0.0, 1.0)
    def exploration_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.choice(self.actions)
        else:
            #get q values for given state
            q_values = [self.q(state, a) for a in self.actions]

            #get greedy action for current state
            return self.actions[np.argmax(q_values)]

    # greedy policy used for explotation (after training)
    def exploitation_policy(self, state):
        #get q values for given state
            q_values = [self.q(state, a) for a in self.actions]

            #get greedy action for current state
            return self.actions[np.argmax(q_values)]
    
    '''
        Choose a random epsilon from (0, 1)
        Choose a random alpha (learning rate) from (0, 1)
        Choose a random gamma (discount factor) from (0, 1)
    '''
        

    def qLearn(self, episodes, epsilon = 0.1, alpha = 0.1, gamma = 0.9):
        # saves the path of each episode
        all_paths = []
        
        for _ in range(episodes):
             current_state = (self.environment.start_x, self.environment.start_y)
             
             '''
             Maximum number of steps per episode
             The episode ends either when the agent reaches the goal or max_steps = 0
             '''
             max_steps = 100
             # save the path for the episode 
             path_per_episode = [current_state]
             while max_steps > 0 and current_state != (self.environment.goal_x, self.environment.goal_y):
                  max_steps -= 1
                  action = self.exploration_policy(current_state, epsilon)

                  new_state, reward = self.environment.step(current_state, action)
                  
                  # get q value for the greedy action of the next state
                  greedy_next_q = max([self.q(new_state, a) for a in self.actions])  

                  # update q value accordingly to newly observed reward
                  self.q_table[current_state, action] = self.q(current_state, action) + alpha * (reward + (gamma * greedy_next_q) - self.q(current_state, action)) 
                  current_state = new_state
                  path_per_episode.append(current_state)

             all_paths.append(path_per_episode)     
        return all_paths
    

    def get_best_path(self, max_steps=100):
        current_state = (self.environment.start_x, self.environment.start_y)
        best_path = [current_state]
        visited = set()

        while current_state != (self.environment.goal_x, self.environment.goal_y):
            if max_steps == 0:
                break
            
            visited.add(current_state)
            action = self.exploitation_policy(current_state)
            next_state, reward = self.environment.step(current_state, action)

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
