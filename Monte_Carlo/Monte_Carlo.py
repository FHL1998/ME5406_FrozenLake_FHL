import math
import random
import time
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from Environment import Environment
from GUI import GUI
from utils.Parameters import *
from utils.Utils import q_table_max_value_dict, plot_final_results, calculate_standard_deviation, calculate_time

random.seed(0)


class Monte_Carlo(object):
    def __init__(self, env, gamma, epsilon):
        # Variable initialization
        self.env = env
        self.NUM_STATES = ENV_HEIGHT * ENV_WIDTH
        self.NUM_ACTION = ACTIONS_NUMBER
        self.ACTION_LIST = list(range(ACTIONS_NUMBER))
        self.NUM_STEPS = NUM_STEPS
        self.epsilon = epsilon
        self.gamma = gamma

        # initialize the episodes number of fail and success within training episodes
        self.success_count, self.fail_count = 0, 0

        # initialize the number of success within training episodes
        self.success_num, self.overall_success_rate = 0, 0

        # initialize the step length within per episode,initialize a list to store the steps.
        self.step = 0
        self.steps = []

        # initialize a dict whose key is the episode number, whose value is the total length within an episode
        self.Episode_Step = {}
        # Initialize a dict to record the step length for success and fail episode
        self.Success_Step, self.Fail_Step = {}, {}
        self.SUCCESS_RATE = {}
        self.success, self.fail, self.optimal = {}, {}, {}

        # initialize the reward agent gained within per episode,and initialize a list to store the rewards
        self.rewards = 0
        self.Reward_List = []

        # initialize the time consuming per episode
        self.start_time = time.time()
        self.Episode_Time = {}

        # initialize the tables of return, q value and number of the states has been visited
        self.Return, self.Q_table, self.Visit_Num = self.initialize_table()

    def initialize_table(self):
        """
        Initialization
        Returns:
        dict:(Q_Table) Initialize the dictionary for storing the Q values;
        dict:(Return) Initialize the dictionary for storing the total return of the given state and action;
        dict:(Visit_Num) Initialize the dictionary for storing the count of the number of times a state-action pair is visited
        """
        Q_Table, Return, Visit_Num = {}, {}, {}

        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTION):
                Q_Table[state] = [0] * self.NUM_ACTION
                Return[state] = [0] * self.NUM_ACTION
                Visit_Num[state] = [0] * self.NUM_ACTION
        return Q_Table, Return, Visit_Num

    def epsilon_greedy_policy(self, state):
        """Usage of epsilon greedy policy to balance exploration and exploitation.

        Args:
            state (int): index coordinate of states

        Returns:
            action(int): if the generalized random value > epsilon, execute optimal action,
                         if the generalized random value < epsilon, choose action randomly.
        """
        # Set the epsilon value(hyper-parameter)
        if random.uniform(0, 1) > self.epsilon:
            actions_values = self.Q_table[state]
            action = actions_values.index(max(actions_values))
        else:
            action = np.random.choice(self.ACTION_LIST)  # Non-greedy mode randomly selects action
        return action

    def generate_episode(self, env, episode):
        """Generate a single episode in First Meet Monte Carlo method
        Start with an init position, and play the episodes while record the info till termination.

        Args:
            episode (int): The episode number during training .

        Returns:
            state_list (list):  Every state tuple for an episode.
            action_list (list):  Every action number taken.
            return_list (list):  Every return for each state-action pair.
        """
        step = 0
        done, success_flag, fail_flag = False, False, False
        observation_state = self.env.reset()
        state_list, action_list, reward_list = [], [], []  # one transition with(s,a,r,t)
        while not done:
            # self.env.render() # The first place should be uncommented if you want to use the GUI

            step += 1

            # choose an action according to epsilon greedy policy
            action = self.epsilon_greedy_policy(observation_state)

            # append the current state, action and reward into the corresponding list
            state_list.append(observation_state)
            action_list.append(action)

            # after the action is made, gain the changes like next_state etc, reward, success_flag, fail_flag.
            observation_state, reward, done, success_flag, fail_flag = self.env.step(action)
            reward_list.append(reward)

            # The episode will end if the agent is at terminal states(ice hole and frisbee)
            if done:
                self.steps += [step]  # Record the step
                self.Episode_Step[episode] = step
                self.current_time = time.time()
                if success_flag:
                    self.start_time, self.Episode_Time = calculate_time(self.current_time, self.start_time, episode)
                    self.success_count += 1  # The number is used to calculate the overall success rate later
                    self.success_num += 1  # The number is used as value of SUCCESS_RATE per 20 episodes
                    self.Success_Step[episode] = step
                    self.success = {k: v for k, v in self.Success_Step.items()}
                    self.optimal = {k: v for k, v in self.Success_Step.items() if v == self.env.shortest}
                elif fail_flag:
                    self.fail_count += 1
                    self.Fail_Step[episode] = step
                    self.fail = {k: v for k, v in self.Fail_Step.items()}
                self.rewards += reward  # record total rewards to calculate average rewards
                self.Reward_List += [self.rewards / (episode + 1)]  # calculate the average reward per episode

            # break the loop if exceed the max step length, this is used as an optimization of Monte Carlo Method
            if self.step > NUM_STEPS:
                break
        self.overall_success_rate = round(self.success_count / (self.success_count + self.fail_count), 2)
        return state_list, action_list, reward_list

    def run(self):
        """The main function to realize the FV Monte Carlo.
        Update the policy table, Q table and return table.

        Returns:
            Q_table (dict):  The complete q table.
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
        # The re-initialization at the start of training
        Q_Value_Per_Episodes = {}  # The q value at certain state
        Q_Value_Diff = {}  # The difference between the q value at certain state and the average q value
        Q_Value_Per_Episodes_Max = {}  # The optimal action value at each state in the episodes
        convergence_list = []  # the number episodes that convergent
        q_convergence_list = []  # the list that store the mean square error value
        min_episode = 0  # the minimum episode number where start to convergent
        standard_deviation = 0  # the standard deviation of action value at certain state

        for episode in range(NUM_EPISODES):  # loop for each episode
            # generate a whole episode and return state list, action list
            state_list, action_list, reward_list = self.generate_episode(self.env, episode)

            G = 0  # initialize the cumulative return

            for i in range(len(state_list) - 1, -1, -1):  # trace back the episode t= T-1,T-2,..,0
                state, action = state_list[i], action_list[i]
                G = self.gamma * G + reward_list[i]  # calculate the discount return

                # judge whether the state is first meet by reverse traversing of the state list
                if state not in state_list[:i]:

                    # update the cumulative return of the state-action pair
                    self.Return[state][action] += G

                    # update the number of times the state-action pair is visited
                    self.Visit_Num[state][action] += 1

                    # calculate the average Q value for each state-action pair by total return divide numbers of visit
                    # Q[St][At]<- average(Return[St][At])
                    self.Q_table[state][action] = self.Return[state][action]/self.Visit_Num[state][action]

            if episode != 0 and episode % 20 == 0:
                success_rate = self.success_num / 20
                self.SUCCESS_RATE[episode] = success_rate
                self.success_num = 0

            Q_State_Values, Max_Q_State_Values = q_table_max_value_dict(self.Q_table)
            Q_Table_Dataframe, Q_Value_Diff, standard_deviation = calculate_standard_deviation(Q_Value_Per_Episodes,
                                                                                               Q_State_Values,
                                                                                               Q_Value_Per_Episodes_Max,
                                                                                               Q_Value_Diff,
                                                                                               episode)

            # judge the convergence of the action value at the last state before frisbee with tolerance 1e-3
            if 0 < standard_deviation < 1e-3:
                convergence_list.append(episode)
                convergence_list_unique = np.unique(convergence_list)
                min_episode = min(convergence_list_unique)

            mse_difference = max((np.sum(np.power(Q_Table_Dataframe, 2))) / (Q_Table_Dataframe.shape[0] * Q_Table_Dataframe.shape[1]))
            q_convergence_list.append(mse_difference)
            print("episode number:{}".format(episode))
        self.env.final_states()
        # print('Q TABLE', self.Q_table) # uncomment this code if you want inspect the q table to debug
        return self.Q_table, self.fail, self.success, self.optimal, self.Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, self.steps, self.SUCCESS_RATE, self.overall_success_rate, self.Reward_List

    def generate_heatmap(self, METHOD_PATH):
        """ This function is used to plot the heatmap of the optimal action value.
        The first step is to gain the full q table. The second step is to gain the optimal action value at each state.
        The third step is to convert the q table into a square matrix whose size equals to grid size and then plot."""
        global ax
        final_route = list(env.final().values())
        final_route.insert(0, (0, 0))  # insert object before index
        Q_State_Values, Max_Q_State_Values = q_table_max_value_dict(self.Q_table)
        Q_Matrix = np.matrix(list(Max_Q_State_Values.values())).reshape(GRID_SIZE, GRID_SIZE)
        sns.axes_style({'axes.edgecolor': 'black'})
        plt.figure(dpi=300)
        plt.rc('font', family='Times New Roman', size=12)
        if GRID_SIZE == 4:
            ax = sns.heatmap(Q_Matrix, mask=None, linewidths=.5, cmap="RdBu_r", annot=True, fmt='.2',
                             vmin=-0.6, vmax=1, annot_kws={'size': 13})
        elif GRID_SIZE == 10:
            ax = sns.heatmap(Q_Matrix, mask=MASK, linewidths=.3, cmap="RdBu_r", annot=False,
                             linecolor='black', vmin=-0.6, vmax=1, annot_kws={'size': 13})
        for rect in final_route:
            print(rect)
            ax.add_patch(Rectangle(rect, 1, 1, fill=False, edgecolor='red', lw=1.8))
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=0, horizontalalignment='right')
        plt.title('Optimal Action Value(FV Monte Carlo)')
        plt.show()
        ax.get_figure().savefig('results/{}/{}/heatmap.jpg'.format(METHOD_PATH, MAP_SIZE), dpi=300, bbox_inches='tight',
                                pad_inches=0.2)


if __name__ == '__main__':
    env = Environment()  # generate the FrozenLake environment
    # env = GUI() # The second place should be uncommented if you want to use the GUI
    METHOD_NAME = 'Monte Carlo'
    METHOD_PATH = 'Monte_Carlo'
    SMOOTH = SMOOTH_SIZE
    Monte_Carlo = Monte_Carlo(env, gamma=GAMMA, epsilon=EPSILON)
    q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List = Monte_Carlo.run()
    Monte_Carlo.generate_heatmap(METHOD_PATH)
    plot_final_results(METHOD_NAME, METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff,
                       min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List)
    # env.mainloop() # The third place should be uncommented if you want to use the GUI
