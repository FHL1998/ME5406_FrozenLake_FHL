# Importing libraries
import tkinter as tk  # To build GUI

import numpy as np

from utils.Parameters import *

# Call and emphasize variables such as pixels, grid size
pixels = PIXELS  # pixels used in GUI
grid_size = GRID_SIZE
env_height = ENV_HEIGHT  # grid height
env_width = ENV_HEIGHT  # grid width

# Global variable for dictionary with coordinates for the final route
final_route = {}


class GUI(tk.Tk, object):
    """Create Environment class for GUI."""
    def __init__(self):
        super(GUI, self).__init__()
        self.n_actions = len(ACTION_SPACE)
        self.canvas_widget = tk.Canvas(self, bg='white', height=env_height * pixels, width=env_width * pixels)
        self.agent_image = ImageTk.PhotoImage(img_agent_image)  # allocate an image of agent in GUI
        self.ice_hole_img = ImageTk.PhotoImage(img_ice_hole)  # allocate images of ice holes in GUI
        self.frisbee_img = ImageTk.PhotoImage(img_frisbee)  # allocate image of frisbee in GUI

        # based on the image allocated, create the elements in the GUI with corresponding positions
        self.frisbee = self.canvas_widget.create_image(pixels * x_frisbee, pixels * y_frisbee, anchor='nw',
                                                       image=self.frisbee_img)
        self.agent = self.canvas_widget.create_image(pixels * x_start, pixels * y_start, anchor='nw',
                                                     image=self.agent_image)

        self.final_route_temp = {}  # dict to store the final route
        self.final_route = {}  # dict to store the final route
        self.c = True

        self.i = 0  # # Key for self.final_route_temp, the number will increase when agent take a step
        self.shortest = 0  # Initialize the steps for the shortest route
        self.ice_holes_positions = []  # Store the obstacles'(ice holes) position
        self.goal_position = None  # Store the goal's position
        self.build_environment()  # call the function to create a environment which includes all settings

    def initial_environment(self):
        """The function aims to create overall arrangement of grid world such as square and lines.
        """
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))

        # creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')

    def build_environment(self):
        """The function aims to build the environment of the GUI by creating the grid world and relevant elements.
        """

        # call the function to generate a square grid world with specific size in Tkinter
        self.initial_environment()

        # Arranging ice holes in the GUI according to positions of ice holes
        for i in range(len(ICE_HOLE)):
            ice_hole_coordinates = ICE_HOLE[i]
            x_ice = ice_hole_coordinates[0]
            y_ice = ice_hole_coordinates[1]
            self.ice_holes = self.canvas_widget.create_image(pixels * x_ice, pixels * y_ice, anchor='nw',
                                                             image=self.ice_hole_img)
            self.ice_holes_positions += [self.canvas_widget.coords(self.ice_holes)]
        self.canvas_widget.pack()  # Packing overall settings and elements into teh environment

    def reset(self):
        """Rest when agent reach the terminal states(fail if encounter ice holes, success if reach the frisbee).
        The initial env will be activated and a new episode will start afterwards.

        Returns:
        observation_state (int): initialize the agent's observation state to start point
        """
        self.update()
        # time.sleep(0.1) # uncomment this if you want to see a more clear movement of the agent

        # delete the agent's icon and reset the agent's position to start point in the GUI
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_image(pixels * x_start, pixels * y_start, anchor='nw',
                                                     image=self.agent_image)

        # Initialize the dict final_route_temp and the i
        self.final_route_temp = {}
        self.i = 0

        # The observation state of the agent is reset to the start point
        observation_state = self.canvas_widget.coords(self.agent)
        observation_state = self.state_to_index(observation_state[0], observation_state[1])
        return observation_state

    def step(self, action):
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
        state = self.canvas_widget.coords(self.agent)  # current state of the agent

        # This array is used to record the change of the coordinate values after the agent take the action.
        base_pixel_coordinate = np.array([0, 0])

        # if the agent take the action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_pixel_coordinate[1] -= pixels  # base_pixel_coordinate[1] =  base_pixel_coordinate[1] - pixels
            else:
                base_pixel_coordinate = base_pixel_coordinate

        # if the agent take the action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_pixel_coordinate[1] += pixels
            else:
                base_pixel_coordinate = base_pixel_coordinate

        # if the agent take the action 'left'
        elif action == 2:
            if state[0] >= pixels:
                base_pixel_coordinate[0] -= pixels
            else:
                base_pixel_coordinate = base_pixel_coordinate

        # if the agent take the action 'right'
        elif action == 3:
            if state[0] < (env_width - 1) * pixels:
                base_pixel_coordinate[0] += pixels
            else:
                base_pixel_coordinate = base_pixel_coordinate

        # Assign the movement to the agent according to the action along x axis and y axis
        # The observation state is calculated by 'adding' and is converted into index coordinates
        self.canvas_widget.move(self.agent, base_pixel_coordinate[0], base_pixel_coordinate[1])
        self.final_route_temp[self.i] = self.canvas_widget.coords(self.agent)
        next_state = self.final_route_temp[self.i]
        self.i += 1

        # Calculating the reward for the agent
        if next_state == self.canvas_widget.coords(self.frisbee):
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

                # Initialize the dict for storing the final route
                self.final_route = {}

                # Assign the values in temporary dict to final route dict
                for j in range(len(self.final_route_temp)):
                    self.final_route[j] = self.final_route_temp[j]

        elif next_state in self.ice_holes_positions:
            reward = -1
            done = True
            fail_flag = True
            self.i = 0

        else:
            reward = 0
            done = False
        next_state = self.state_to_index(next_state[0], next_state[1])

        return next_state, reward, done, success_flag, fail_flag

    def render(self):
        """The function is used to update and refresh the global change in the environment.
        """
        # time.sleep(0.05)  # uncomment this code if you want to observe the detailed movement of agent
        self.update()

    # Function to show the found route
    def final(self):
        """The function is used to gain the final route of the agent based on the shortest route
        and illustrate the route in the canvas.
        """

        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)

        # showing the initial state of the agent in the GUI
        origin = np.array([20, 20])
        self.initial_point = self.canvas_widget.create_oval(
            origin[0] - 5, origin[1] - 5,
            origin[0] + 5, origin[1] + 5,
            fill='red', outline='red')

        #  showing the final route with dot in the center of each square
        for j in range(len(self.final_route)):
            self.track = self.canvas_widget.create_oval(
                self.final_route[j][0] + origin[0] - 5, self.final_route[j][1] + origin[0] - 5,
                self.final_route[j][0] + origin[0] + 5, self.final_route[j][1] + origin[0] + 5,
                fill='red', outline='red')
            final_route[j] = self.final_route[j]
        return final_route

    def state_to_index(self, x, y):
        """Transfer the coordinate(x,y) into the index using (x+env_width*y).
        Note: for square env, env_width= env_height=env_size

        Args:
            x (int): x coordinate of states in tuple(x,y)
            y (int): y coordinate of states in tuple(x,y)
        Returns:
            state_index (int): the index coordinate
        """
        state_index = int(x / PIXELS) + int(y / PIXELS * GRID_SIZE)
        return state_index

    def final_states(self):
        return self.final_route


# In order to show the interface without running the entire program, every file has a main function to debug
if __name__ == '__main__':
    env = GUI()
    env.mainloop()
