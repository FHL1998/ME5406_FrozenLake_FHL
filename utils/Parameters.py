from PIL import Image, ImageTk  # For adding images into the canvas widget
from map.map_process import transfer_matrix_to_coordinate  # please refer to map_process for more details


'''The first place need to be changed if you want to execute the methods in different map size'''
# Environment parameters
MAP_NAME = 'map/map_10x10.txt'  # MAP_NAME could be selected within 'map/map_10x10.txt' and 'map/map_4x4.txt'
MAP_SIZE = '10x10'  # MAP_SIZE used in plot, should in line with MAP_NAME, has options of '10x10' and '4x4'


'''The second place need to be changed if you want to execute the methods with different hyper-parameters'''
# Training parameters
# NUM_STEPS restrict the maximum steps for Monte Carlo Method in single episode
NUM_STEPS = 150
# NUM_EPISODES is the episodes that involved in training process
NUM_EPISODES = 3000
# LEARNING_RATE is the hyper-parameter used in Q-Learning, SARSA and SARSA(lambda)
LEARNING_RATE = 0.1
# GAMMA is the hyper-parameter illustrate the discount rate of return
GAMMA = 0.95
# EPSILON is the hyper-parameter illustrate the relationship between greedy and exploration
EPSILON = 0.1
# LAMBDA is the hyper-parameter used in SARSA(lambda) only, do not change this in other methods.
LAMBDA = 0.9


''' The third place you should modify if you want to execute different tasks. Change the OVERALL_TASK first and if the 
OVERALL_TASK is Tuning Q Learning or Tuning SARSA, you can change the specific tuning tasks in TASK.'''

# OVERALL_TASK_LIST is the list involved in the 3 tasks, and TASK is the current task
OVERALL_TASK_LIST = ['Run Three Methods', 'Compare Three Methods', 'Tuning Q Learning', 'Tuning SARSA']
OVERALL_TASK = 'Tuning Q Learning'

# TUNING_TASK_LIST is the list involved in 3 tasks when the OVERALL_TASK is 'Tuning Q Learning' or 'Tuning SARSA',
# and TASK is the current tuning task
TUNING_TASK_LIST = ['Tuning Learning Rate', 'Tuning Discount Rate', 'Tuning Greedy Policy']
TASK = 'Tuning Learning Rate'


'''These parameters are relative to environment and are fixed setting or will be automatically generated, 
so there is no modification need.'''

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

'''Figure plot relative parameters, 10 for default smooth parameter, 
change it only if you want to achieve a smoother curve.'''
# SMOOTH_SIZE reveals the window size in the moving average method to smooth the figure
SMOOTH_SIZE = 10
