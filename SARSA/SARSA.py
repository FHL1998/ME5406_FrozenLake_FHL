import random
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from Environment import Environment
from GUI import GUI
from utils.Parameters import *
from utils.Utils import calculate_standard_deviation, calculate_time, q_table_max_value_dict, plot_final_results

random.seed(0)


class SARSA(object):
    """Create SARSA class."""
    def __init__(self, env, learning_rate, gamma, epsilon):
        self.env = env
        self.actions = list(range(ACTIONS_NUMBER))
        self.METHOD_NAME = SARSA
        self.NUM_STATES = ENV_HEIGHT * ENV_WIDTH
        self.NUM_ACTIONS = ACTIONS_NUMBER
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_Table = self.initialize_table()

    def initialize_table(self):
        """
        Initialize tables with certain data structure.

        Returns:
            Q_table (dict):  A dict mapping every state to action score list. Init with 0.
            For example: {0: [0, 0, 0, 0], 1: [0, 0, 0, 0], ...}
        """
        self.Q_Table = {}
        for state in range(self.NUM_STATES):
            state = state  # state is a int that represents index coordinates ,like 0,1,2
            self.Q_Table[state] = [0] * self.NUM_ACTIONS
        return self.Q_Table

    def epsilon_greedy(self, state):
        """Usage of epsilon greedy policy to balance exploration and exploitation.

        Args:
            state (str): index coordinate of states

        Returns:
            action(int): if the generalized random value > epsilon, execute optimal action,
                         if the generalized random value < epsilon, choose action randomly.
        """
        if random.uniform(0, 1) > self.epsilon:
            actions_values = self.Q_Table[state]
            action = actions_values.index(max(actions_values))
        else:
            action = np.random.choice(self.actions)  # Non-greedy mode randomly selects action
        return action

    def learn(self, current_state, current_action, reward, next_state, next_action):
        """The update manner of SARSA to update action value at each state based on the pair(st,at,rt,st+1, at+1)
        Specifically, the update policy is: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        Unlike Q Learning, the action SARSA select at next observation state based on epsilon-greedy rather than greedy,
        Which means there still exists possibility for SARSA to random choose the action even achieve optimal policy.

        Args:
            current_state (int): The observation state of the agent.
            current_action (int): The action selected based on the epsilon-greedy policy at current state
            reward (int): The reward gained by taking the action
            next_state (int): The agent's observation state after implement the action
            next_action (int): The action selected based on the epsilon-greedy policy at next state
        Returns:
            q value: q value in a given state and action
        """
        Q_Predict = self.Q_Table[current_state][current_action]
        TD_target = reward + self.gamma * self.Q_Table[next_state][next_action]
        TD_Error = TD_target - Q_Predict
        self.Q_Table[current_state][current_action] += self.learning_rate * TD_Error
        return self.Q_Table[current_state][current_action]  # return Q value in a given state and action

    def train(self, num_epoch):
        """The main function to realize the SARSA. Update the Q table and get the relevant performance.

        Returns:
            Q_table (dataframe):  The complete q table.
            fail (dict): The dict that stores the episode as key, and length of steps in an fail episode as value.
            success (dict): The dict that stores the episode as key, and length of steps in an success episode as value.
            optimal (dict): The dict that stores the episode as key, and length of steps using optimal policy as value.
            Episode_Time (dict): The dict that stores the episode as key, and time consumed as value.
            Q_Value_Diff (dict): The dict that stores the episode as key, and time consumed as value.
            min_episode (int) : The minimum episode that state value convergent(with a tolerance of 1e-3).
            q_convergence_list (list) : The list is used to store all episode number that convergent.
            steps(list) : The list that stores the step length in an episode
            SUCCESS_RATE(dict) : The dict that stores the episode as key, and success rate per episode as value.
            overall_success_rate(int) : Illustrate the overall success rate during the training process
            Reward_List(list): The list that stores the average rewards
        """

        global Q_Value_Per_Episodes, Q_Value_Per_Episodes_Max, Q_Value_Diff, convergence_list, \
            q_convergence_list, standard_deviation, min_episode

        # initialize the number of success and overall success rate within training episodes
        success_count, fail_count = 0, 0

        # initialize the number of success within training episodes
        success_num, overall_success_rate = 0, 0

        # initialize a list to store the steps.
        steps = []

        # initialize a dict whose key is the episode number, whose value is the total length within an episode
        Episode_Step = {}
        # Initialize a dict to record the step length for success and fail episode
        Success_Step, Fail_Step = {}, {}
        SUCCESS_RATE = {}
        success, fail, optimal = {}, {}, {}

        # initialize the reward agent gained within per episode,and initialize a list to store the rewards
        rewards = 0
        Reward_List = []

        # initialize the time consuming per episode
        start_time = time.time()
        Episode_Time = {}

        for i in range(num_epoch):
            current_state = self.env.reset()  # reset the environment and get initial observation state

            # current action selection based on epsilon greedy policy at current observation state
            current_action = self.epsilon_greedy(current_state)

            step = 0

            #  The re-initialization at the start of training
            if i == 0:
                Q_Value_Per_Episodes = {}  # The q value at certain state
                Q_Value_Diff = {}  # The difference between the q value at certain state and the average q value
                Q_Value_Per_Episodes_Max = {}  # The optimal action value at each state in the episodes
                convergence_list = []  # the number episodes that convergent
                q_convergence_list = []  # the list that store the mean square error value
                min_episode = 0  # the minimum episode number where start to convergent
                standard_deviation = 0  # the standard deviation of action value at certain state
                self.Q_Table = self.initialize_table()

            # Calculate the success rate per 20 episodes
            if i != 0 and i % 20 == 0:
                success_rate = success_num / 20
                SUCCESS_RATE[i] = success_rate
                success_num = 0

            Q_State_Values, Max_Q_State_Values = q_table_max_value_dict(self.Q_Table)
            Q_Table_Dataframe, Q_Value_Diff, standard_deviation = calculate_standard_deviation(Q_Value_Per_Episodes,
                                                                                               Q_State_Values,
                                                                                               Q_Value_Per_Episodes_Max,
                                                                                               Q_Value_Diff,
                                                                                               i)

            # judge the convergence of the action value at the last state before frisbee with tolerance 1e-3
            if 0 < standard_deviation < 1e-3:
                convergence_list.append(i)
                convergence_list_unique = np.unique(convergence_list)
                min_episode = min(convergence_list_unique)

            mse_difference = max((np.sum(np.power(Q_Table_Dataframe, 2))) / (Q_Table_Dataframe.shape[0] * Q_Table_Dataframe.shape[1]))
            q_convergence_list.append(mse_difference)

            while True:
                # self.env.render()  # The first place should be uncommented if you want to use the GUI

                # after the action is made, gain the changes like next_state etc, reward, success_flag, fail_flag.
                next_state, reward, done, success_flag, fail_flag = self.env.step(current_action)

                # next action selection based on epsilon greedy policy at next observation state
                next_action = self.epsilon_greedy(next_state)

                # update the action value of each state by SARSA with pair (st,at,rt,st+1, at+1)
                self.learn(current_state, current_action, reward, next_state, next_action)
                step += 1
                if done:
                    steps += [step]  # Record the step
                    Episode_Step[i] = step
                    current_time = time.time()
                    if success_flag:
                        start_time, Episode_Time = calculate_time(current_time, start_time, i)
                        success_count += 1
                        success_num += 1
                        Success_Step[i] = step
                        success = {k: v for k, v in Success_Step.items()}
                        optimal = {k: v for k, v in Success_Step.items() if v == self.env.shortest}
                    else:
                        fail_count += 1
                        Fail_Step[i] = step
                        fail = {k: v for k, v in Fail_Step.items()}

                    rewards += reward  # record total rewards to calculate average rewards per episode
                    Reward_List += [rewards / (i + 1)]  # calculate the average reward per episode
                    break

                # Swapping the observation states and the observation state before taking action
                current_state = next_state

                # Swapping the next action and the current action
                current_action = next_action

            print("episodes number:{}".format(i))

        # print('Q TABLE', self.Q_Table) # uncomment this code if you want inspect the q table to debug
        self.env.final()

        # calculate the overall success rate based on the success number and fail number
        overall_success_rate = round(success_count / (success_count + fail_count), 2)
        return self.Q_Table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, \
               steps, SUCCESS_RATE, overall_success_rate, Reward_List

    def generate_heatmap(self,METHOD_NAME):
        """ This function is used to plot the heatmap of the optimal action value.
        The first step is to gain the full q table. The second step is to gain the optimal action value at each state.
        The third step is to convert the q table into a square matrix whose size equals to grid size and then plot."""
        global ax
        final_route_values_list = list(self.env.final().values())
        final_route_values_list.insert(0, (0, 0))
        Q_State_Values, Max_Q_State_Values = q_table_max_value_dict(self.Q_Table)
        Q_Matrix = np.matrix(list(Max_Q_State_Values.values())).reshape(GRID_SIZE, GRID_SIZE)
        sns.axes_style({'axes.edgecolor': 'black'})
        plt.rc('font', family='Times New Roman', size=12)
        if GRID_SIZE == 4:
            ax = sns.heatmap(Q_Matrix, mask=None, linewidths=.5, cmap="RdBu_r", annot=True, fmt='.2',
                             vmin=-0.6, vmax=1, annot_kws={'size': 13})
        elif GRID_SIZE == 10:
            ax = sns.heatmap(Q_Matrix, mask=MASK, linewidths=.3, cmap="RdBu_r", annot=False,
                             linecolor='black', vmin=-0.6, vmax=1, annot_kws={'size': 13})
        for rect in final_route_values_list:
            ax.add_patch(Rectangle(rect, 1, 1, fill=False, edgecolor='red', lw=2))
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=0, horizontalalignment='right')  # 设置y轴的坐标排列方式
        plt.title('Maximum Action Value(SARSA)')
        plt.show()
        ax.get_figure().savefig('results/{}/{}/heatmap.jpg'.format(METHOD_NAME, MAP_SIZE), dpi=300,
                                bbox_inches='tight', pad_inches=0.2)


if __name__ == '__main__':
    METHOD_NAME = 'SARSA'
    METHOD_PATH = 'SARSA'
    SMOOTH = SMOOTH_SIZE
    # env = GUI()  # The second place should be uncommented if you want to use the GUI
    env = Environment()  # create a FrozenLake environment
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, \
    overall_success_rate, Reward_List = SARSA.train(num_epoch=NUM_EPISODES)
    plot_final_results(METHOD_NAME,METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode,
                       q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List)
    SARSA.generate_heatmap(METHOD_NAME)
    # env.mainloop()  # The third place should be uncommented if you want to use the GUI
