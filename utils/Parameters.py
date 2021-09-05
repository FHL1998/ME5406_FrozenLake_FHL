from PIL import Image, ImageTk  # For adding images into the canvas widget
from map.map_process import transfer_matrix_to_coordinate  # please refer to map_process for more details

# Environment parameters
MAP_NAME = 'map/map_4x4.txt'  # MAP_NAME could be selected within 'map/map_10x10.txt' and 'map/map_4x4.txt'
MAP_SIZE = '4x4'  # MAP_SIZE used in plot, should in line with MAP_NAME, has options of '10x10' and '4x4'
GRID_SIZE, map_matrix, START, ICE_HOLE, FRISBEE, MASK = transfer_matrix_to_coordinate(MAP_NAME)
PIXELS = 40
x_start = START[0][0]
y_start = START[0][1]
x_frisbee = FRISBEE[0][0]
y_frisbee = FRISBEE[0][1]

img_agent_image = Image.open("images/agent.png")  # The image of agent in tkinter GUI
img_ice_hole = Image.open("images/ice.png")  # The image of frisbee in tkinter GUI
img_frisbee = Image.open("images/home.png")  # The image of ice holes in tkinter GUI

ENV_HEIGHT = GRID_SIZE  # grid height
ENV_WIDTH = GRID_SIZE  # grid width
ACTION_SPACE = ['up', 'down', 'left', 'right']  # list that stores all possible actions of agent
ACTIONS_NUMBER = len(ACTION_SPACE)  # (int):the length of the action space
STATES_SPACE = ENV_HEIGHT * ENV_WIDTH  # (int): the total possible states in the environment

# Training parameters
# NUM_STEPS restrict the maximum steps for Monte Carlo Method in single episode
NUM_STEPS = 150
# NUM_EPISODES is the episodes that involved in training process
NUM_EPISODES = 1000
# LEARNING_RATE is the hyper-parameter used in Q-Learning, SARSA and SARSA(lambda)
LEARNING_RATE = 0.1
# GAMMA is the hyper-parameter illustrate the discount rate of return
GAMMA = 0.95
# EPSILON is the hyper-parameter illustrate the relationship between greedy and exploration
EPSILON = 0.1
# LAMBDA is the hyper-parameter used in SARSA(lambda) only
LAMBDA = 0.9

# Figure plot relative parameters
# SMOOTH_SIZE reveals the window size in the moving average method to smooth the figure
SMOOTH_SIZE = 10
# TASK_LIST is the list involved in the 3 tasks, and TASK is the current task
TASK_LIST = ['Tuning Learning Rate', 'Tuning Discount Rate', 'Tuning Greedy Policy']
TASK = 'Tuning Greedy Policy'
