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
        goal = (self.goal_x, self.goal_y)

        # Calculate intended new state based on action
        if action == 'up' and y < self.height - 1:
            intended_state = (x, y + 1)
        elif action == 'down' and y > 0:
            intended_state = (x, y - 1)
        elif action == 'right' and x < self.width - 1:
            intended_state = (x + 1, y)
        elif action == 'left' and x > 0:
            intended_state = (x - 1, y)
        else:
            intended_state = (x, y)

        # If intended state is obstacle, stay in place and apply penalty
        if intended_state in self.obstacles:
            new_state = (x, y)
            reward = -50  # Obstacle penalty
        elif intended_state == goal:
            new_state = intended_state
            reward = 100  # Goal reward
        else:
            new_state = intended_state
            reward = -1  # Default move penalty

        return new_state, reward