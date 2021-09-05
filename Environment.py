# Importing libraries
import numpy as np  # To deal with data in form of matrices
# Importing gloabal parameters
from utils.Parameters import *


# Call and emphasize variables such as grid size
grid_size = GRID_SIZE
env_height = ENV_HEIGHT  # grid height
env_width = ENV_HEIGHT  # grid width

# Global variable for dictionary with coordinates for the final route
global min_steps_route


# Creating class for the environment
class Environment:
    """Create Environment class ."""
    def __init__(self):
        super(Environment, self).__init__()
        self.NUM_ACTIONS = len(ACTION_SPACE)
        self.frisbee = [(x_frisbee, y_frisbee)]
        self.agent = (x_start, y_start)
        self.ice_holes = ICE_HOLE

        self.routes = {}
        self.final_route_temp = {}  # dict to store the final route
        self.final_route = {}  # dict to store the final route
        self.FINAL_ROUTE = {}  # dict to store the final route
        self.c = True

        self.i = 0  # Key for self.final_route_temp, the number will increase when agent take a step
        self.shortest = 0  # Initialize the steps for the shortest route
        self.ice_holes_positions = []  # Store the obstacles'(ice holes) position
        self.goal_position = None  # Store the goal's position

    def reset(self):
        """Rest when agent reach the terminal states(fail if encounter ice holes, success if reach the frisbee).

        Returns:
        current_state (int): initialize the agent's current state to start point
        """
        self.final_route_temp = {}
        self.i = 0
        self.agent = (x_start, y_start)
        # current_state = (self.agent[0], self.agent[1])
        current_state = self.state_to_index(self.agent[0], self.agent[1])
        return current_state

    def step(self, action: int):
        """Take a step given action, and return the observation state, reward, and corresponding flag.
        The agent will stay still is the action will lead it exceed the boundary of environment.

        Args:
            action (int): input an action, which is chose from the index of ['up', 'down', 'left', 'right']

        Returns:
            next_state (int): states in index form.
            reward (int): reward gained in one single step.
            done (boolean): the episode is terminated when agent reach the frisbee or fall into ice holes.
            success_flag (boolean): is True when agent reach frisbee in the episode.
            fail_flag (boolean): is True when agent fall into ice holes.
        """
        success_flag = False
        fail_flag = False
        state = self.agent  # current state of the agent

        # This array is used to record the change of the coordinate values after the agent take the action.
        base_coordinate = np.array([0, 0])

        # if the agent take the action 'up'
        if action == 0:
            if state[1] >= 1:
                base_coordinate[1] -= 1
            else:
                base_coordinate = base_coordinate

        # if the agent take the action 'down'
        elif action == 1:
            if state[1] < (env_height - 1):
                base_coordinate[1] += 1
            else:
                base_coordinate = base_coordinate

        # if the agent take the action 'left'
        elif action == 2:
            if state[0] >= 1:
                base_coordinate[0] -= 1
            else:
                base_coordinate = base_coordinate

        # if the agent take the action 'right'
        elif action == 3:
            if state[0] < (env_width - 1):
                base_coordinate[0] += 1
            else:
                base_coordinate = base_coordinate

        # updating the agent's state caused by action took
        self.agent = (self.agent[0] + base_coordinate[0], self.agent[1] + base_coordinate[1])
        self.final_route_temp[self.i] = self.agent
        observation_state = self.final_route_temp[self.i]
        self.i += 1

        # The agent will receive a reward with 1 if reach the frisbee
        if observation_state in self.frisbee:
            reward = 1
            done = True
            success_flag = True
            if self.c:
                for j in range(len(self.final_route_temp)):
                    self.final_route[j] = self.final_route_temp[j]
                self.c = False
                self.shortest = len(self.final_route_temp)

            # Judge whether the temporary route is the shortest route
            if len(self.final_route_temp) < len(self.final_route):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.final_route_temp)

                # Initialize the dict for final route
                self.final_route = {}

                # Assign the values in temporary dict to final route dict
                for j in range(len(self.final_route_temp)):
                    self.final_route[j] = self.final_route_temp[j]

        elif observation_state in self.ice_holes:
            reward = -1
            done = True
            fail_flag = True
            self.i = 0

        else:
            reward = 0
            done = False
        next_state = self.state_to_index(observation_state[0], observation_state[1])
        # next_state = (observation_state[0], observation_state[1])
        return next_state, reward, done, success_flag, fail_flag

    # Function to show the found route
    def final(self):
        """Demonstrate the final route under optimal policy.

        Returns:
            FINAL_ROUTE (dict): the dictionary that store the
        """
        print('The shortest route:', self.shortest)

        # Filling the route
        for j in range(len(self.final_route)):
            self.FINAL_ROUTE[j] = self.final_route[j]
        return self.FINAL_ROUTE

    def state_to_index(self, x: int, y: int) -> int:
        """Transfer the coordinate(x,y) into the index using (x+env_width*y).
        Note: for square env, env_width= env_height=env_size

        Args:
            x (int): x coordinate of states in tuple(x,y)
            y (int): y coordinate of states in tuple(x,y)
        Returns:
            state_index (int): the index coordinate
        """
        state_index = int(x) + int(y * GRID_SIZE)
        return state_index

    def final_states(self):
        return self.final_route
