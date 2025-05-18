class Environment:
    def __init__(self, width, height, start_x, start_y, goal_x, goal_y):
        self.width = width
        self.height = height
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y  
        self.obstacles = set()

    def step(self, state, action):
        x, y = state
        new_x, new_y = x, y

        # Calculate new state based on action
        if action == 'up' and x > 0:
            new_x = x - 1
        elif action == 'down' and x < self.width - 1:
            new_x = x + 1
        elif action == 'left' and y > 0:
            new_y = y - 1
        elif action == 'right' and y < self.height - 1:
            new_y = y + 1

        new_state = (new_x, new_y)

        # Assign rewards
        if new_state == self.goal:
            reward = 100  # Goal reward
        elif new_state in self.obstacles:
            reward = -2   # Obstacle penalty
        else:
            reward = -1   # Default move penalty

        return new_state, reward