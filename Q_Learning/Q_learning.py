import random
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from Environment import Environment
from GUI import GUI
from utils.Parameters import *
from utils.Utils import calculate_standard_deviation_Q_Learning, calculate_time, q_table_to_array, plot_final_results

random.seed(0)


class Q_Learning(object):
    """Create Q Learning class."""

    def __init__(self, env, learning_rate, gamma, epsilon):

        self.env = env
        self.actions = list(range(ACTIONS_NUMBER))  # List of actions [0, 1, 2, 3]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize full Q-table using dataframe
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        # Initialize Q-table dataframe of the final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_validation(self, state):
        """If the observation state is not in the q table, add a new row for the state to create a full q table ."""
        if state not in self.q_table.index:

            # In each state, initialize the Q value of each action to 0
            q_table_new_row = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(q_table_new_row)

    def epsilon_greedy_policy(self, current_state):
        """Usage of epsilon greedy policy to balance exploration and exploitation.

        Args:
            current_state (str): index coordinate of states

        Returns:
            action(int): if the generalized random value > epsilon, execute optimal action,
                         if the generalized random value < epsilon, choose action randomly.
        """
        self.check_state_validation(current_state)
        if np.random.random() > self.epsilon:

            # gain the 4 Q value at current state corresponds to 4 actions
            state_action = self.q_table.loc[current_state, :]
            max_q_value_at_current_state = np.max(state_action)

            # Randomly find the position with the largest column value in a row,
            # and perform the action(the largest column value may be multiple, randomly selected)
            action = np.random.choice(state_action[state_action == max_q_value_at_current_state].index)
        else:
            action = np.random.choice(self.actions)  # Non-greedy mode randomly selects action
        return action

    def learn(self, state, action, reward, next_state):
        """The update manner of Q Learning to update the action value at each state based on the pair(st,at,rt,st+1)
        Specifically, the update formula is: Q(s,a) = Q(s,a) + alpha *(r + gamma * max[Q(s',a)] - Q(s,a))

        Args:
            state (str): The observation state of the agent.
            action (int): The action selected based on the epsilon-greedy policy
            reward (int): The reward gained by taking the action
            next_state (str): The agent's observation state after implement the action
        Returns:
            q value: q value in a given state and action
        """

        # Checking if the next step already exists in the Q-table
        self.check_state_validation(next_state)

        # Calculate the q target value(estimated target value) according to update rules
        # Specifically, the argmax operation is used in Q learning to get the optimal action value at observation state
        TD_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        TD_error = TD_target - self.q_table.loc[state, action]

        # Updating action value in Q-table using TD method
        self.q_table.loc[state, action] += self.learning_rate * TD_error
        return self.q_table.loc[state, action]

    # Train for updating the Q table
    def train(self, num_episode):
        """The main function to realize the Q Learning. Update the Q table and get the relevant performance.

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

        success_count, fail_count = 0, 0

        # initialize the number of success and overall success rate within training episodes
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

        for i in range(num_episode):

            current_state = self.env.reset()
            step = 0  # Initialize step count

            #  The re-initialization at the start of training
            if i == 0:
                Q_Value_Per_Episodes = {}  # The q value at certain state
                Q_Value_Diff = {}  # The difference between the q value at certain state and the average q value
                Q_Value_Per_Episodes_Max = {}  # The optimal action value at each state in the episodes
                convergence_list = []  # the number episodes that convergent
                q_convergence_list = []  # the list that store the mean square error value
                min_episode = 0  # the minimum episode number where start to convergent
                standard_deviation = 0  # the standard deviation of action value at certain state
                self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

            # Calculate the success rate per 20 episodes
            if i != 0 and i % 20 == 0:
                success_rate = success_num / 20
                SUCCESS_RATE[i] = success_rate
                success_num = 0

            Q_Value_Diff, standard_deviation = calculate_standard_deviation_Q_Learning(Q_Value_Per_Episodes,
                                                                                       self.q_table,
                                                                                       Q_Value_Per_Episodes_Max,
                                                                                       Q_Value_Diff,
                                                                                       i)

            # judge the convergence of the action value at the last state before frisbee with tolerance 1e-3
            if 0 < standard_deviation < 1e-3:
                convergence_list.append(i)
                convergence_list_unique = np.unique(convergence_list)
                min_episode = min(convergence_list_unique)

            mse_difference = max((np.sum(np.power(self.q_table, 2))) / (self.q_table.shape[0] * self.q_table.shape[1]))
            q_convergence_list.append(mse_difference)  # Append the MSE value to the q_convergence_list

            while True:
                # self.env.render()  # The first place should be uncommented if you want to use the GUI

                # action selection based on epsilon greedy policy
                action = self.epsilon_greedy_policy(str(current_state))

                # after the action is made, gain the changes like next_state etc, reward, success_flag, fail_flag.
                next_state, reward, done, success_flag, fail_flag = self.env.step(action)

                # update the action value of each state by Q Learning with pair(st,at,rt,st+1)
                self.learn(str(current_state), action, reward, str(next_state))

                # Swapping the current observation states and the observation state before taking action
                current_state = next_state

                step += 1  # Count the number of Steps in the current Episode

                # The episode will end if the agent is at terminal states(ice hole and frisbee)
                if done:
                    steps += [step]  # record the step into the list
                    Episode_Step[i] = step
                    current_time = time.time()

                    # for the case that the agent successfully reach the frisbee
                    if success_flag:
                        start_time, Episode_Time = calculate_time(current_time, start_time, i)
                        success_count += 1  # The number is used to calculate the overall success rate later
                        success_num += 1  # The number is used as value of SUCCESS_RATE per 20 episodes
                        Success_Step[i] = step  # the episode is considered as key while step length as value of dict
                        success = {k: v for k, v in Success_Step.items()}
                        optimal = {k: v for k, v in Success_Step.items() if v == self.env.shortest}

                    # for the case that the agent fall into ice holes
                    else:
                        fail_count += 1
                        Fail_Step[i] = step
                        fail = {k: v for k, v in Fail_Step.items()}

                    rewards += reward  # record total rewards to calculate average rewards per episode
                    Reward_List += [rewards / (i + 1)]  # calculate the average reward per episode
                    break
            print('episode number:{}'.format(i))
        # print('Q TABLE', self.q_table) # uncomment this code if you want inspect the q table to debug
        self.env.final()

        # calculate the overall success rate based on the success number and fail number with two decimal places
        # specifically, it equals to success number/(success number+fail number)
        overall_success_rate = round(success_count / (success_count + fail_count), 2)
        self.print_q_table()
        return self.q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, \
               steps, SUCCESS_RATE, overall_success_rate, Reward_List

    # Printing the Q-table with states
    def print_q_table(self):
        e = self.env.final_states()
        # comparing the indexes with coordinates and writing into the new Q-table values
        for i in range(len(e)):
            state = str(e[i])
            # going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]
        return self.q_table

    def generate_heatmap(self, METHOD_PATH):
        """ This function is used to plot the heatmap of the optimal action value.
        The first step is to gain the full q table. The second step is to gain the optimal action value at each state.
        The third step is to convert the q table into a square matrix whose size equals to grid size and then plot."""
        global ax
        final_route = list(self.env.final().values())
        final_route.insert(0, (0, 0))  # insert object before index
        full_Q_Table_max_array = q_table_to_array(self.q_table)
        sns.axes_style({'axes.edgecolor': 'black'})
        plt.rc('font', family='Times New Roman', size=12)
        if GRID_SIZE == 4:
            ax = sns.heatmap(full_Q_Table_max_array, mask=None, linewidths=.5, cmap="RdBu_r", annot=True, fmt='.2',
                             vmin=-0.6, vmax=1, annot_kws={'size': 13})
        elif GRID_SIZE == 10:
            ax = sns.heatmap(full_Q_Table_max_array, mask=MASK, linewidths=.3, cmap="RdBu_r", annot=False,
                             linecolor='black', vmin=-0.6, vmax=1, annot_kws={'size': 13})
        for rect in final_route:
            ax.add_patch(Rectangle(rect, 1, 1, fill=False, edgecolor='red', lw=1.8))
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=0, horizontalalignment='right')  # 设置y轴的坐标排列方式
        plt.title('Maximum Action Value(Q-Learning)')
        plt.show()
        ax.get_figure().savefig('results/{}/{}/heatmap.jpg'.format(METHOD_PATH, MAP_SIZE), dpi=300, bbox_inches='tight',
                                pad_inches=0.2)


if __name__ == "__main__":
    METHOD_NAME = 'Q Learning'
    METHOD_PATH = 'Q_Learning'
    SMOOTH = SMOOTH_SIZE
    env = Environment()  # generate the FrozenLake environment
    # env = GUI()  # The second place should be uncommented if you want to use the GUI
    Q_Learning = Q_Learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)  # Create a q learning agent
    q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, \
    overall_success_rate, Reward_List = Q_Learning.train(num_episode=NUM_EPISODES)  # Learning and updating
    plot_final_results(METHOD_NAME, METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff,
                       min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List)
    Q_Learning.generate_heatmap(METHOD_PATH)
    # env.mainloop()  # The third place should be uncommented if you want to use the GUI
