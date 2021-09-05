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
from utils.Utils import calculate_standard_deviation, q_table_to_array, calculate_time, plot_diff_values, \
    plot_q_convergence, plot_average_rewards, plot_success_rate, q_table_max_value_dict, plot_final_results

random.seed(0)


# Creating class for the SarsaTable
class SARSA(object):
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
            dict: (Q_table) A dict mapping every state to action score list. Init with 0.
                For example: {(0, 0): [0, 0, 0, 0], (1, 0): [0, 0, 0, 0], ...}
        """
        self.Q_Table = {}
        for state in range(self.NUM_STATES):
            state = state  # state is a binary array,like (0,0),(1,0),(2,0)
            self.Q_Table[state] = [0] * self.NUM_ACTIONS
        return self.Q_Table

    def epsilon_greedy(self, epsilon_schedule, episode, state):
        # Set the epsilon value(hyper-parameter)
        global action
        if epsilon_schedule == 'increase':
            epsilon = 0.1+0.9*((episode+1)/NUM_EPISODES)
            if epsilon > 0.3:
                epsilon = 0.1
            if random.uniform(0, 1) > epsilon:
                actions_values = self.Q_Table[state]
                action = actions_values.index(max(actions_values))
            else:
                action = np.random.choice(self.actions)
        elif epsilon_schedule == 'decrease':
            epsilon = 1-((episode+1)/NUM_EPISODES)
            if epsilon < 0.2:
                epsilon = 0.2
            if random.uniform(0, 1) > epsilon:
                actions_values = self.Q_Table[state]
                action = actions_values.index(max(actions_values))
            else:
                action = np.random.choice(self.actions)

        return action

    # Choose actions based on optimal greedy policy
    def optimal_policy(self, current_state):
        actions_values = self.Q_Table[current_state]
        action = actions_values.index(max(actions_values))
        return action

    def learn(self, current_state, current_action, reward, next_state, next_action):
        Q_Predict = self.Q_Table[current_state][current_action]
        TD_target = reward + self.gamma * self.Q_Table[next_state][next_action]
        TD_Error = TD_target - Q_Predict
        self.Q_Table[current_state][current_action] += self.learning_rate * TD_Error
        return self.Q_Table[current_state][current_action]  # return 给定状态和动作下的Q value

    # Learning and updating the Q table using the SARSA update rules as :
    # Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
    def train(self, epsilon_schedule, num_epoch):
        # Resulted list for the plotting Episodes via Steps
        steps = []
        accuracy = []
        Reward_List = []
        q_value_list = []
        convergence_list = []
        q_convergence_list = []
        Q_Value_Per_Episodes = {}  # dictionary for Q value
        Q_Value_Diff = {}
        Q_Value_Per_Episodes_Max = {}

        Episode_Step = {}
        Episode_Time = {}
        Success_Step = {}
        Fail_Step = {}
        SUCCESS_RATE = {}
        success = {}
        optimal = {}
        fail = {}

        # Initialize the counts
        min_episode = 0
        rewards = 0
        success_num = 0
        success_count = 0
        fail_count = 0
        start_time = time.time()

        # for each episode
        for i in range(num_epoch):
            current_state = self.env.reset()  # reset the environment and get initial current_state
            # if random.uniform(0, 1) < (0.05+((i+1)/NUM_EPISODES)):
            #     actions_values = self.Q_Table[current_state]
            #     current_action = actions_values.index(max(actions_values))
            # else:
            #     current_action = np.random.choice(self.actions)
            current_action = self.epsilon_greedy(epsilon_schedule, i, current_state)
            step = 0
            if i == 0:
                Q_Value_Per_Episodes = {}  # Q_Value_Per_Episodesionary for Q value
                Q_Value_Diff = {}
                Q_Value_Per_Episodes_Max = {}
                convergence_list = []
                q_convergence_list = []
                min_episode = 0
                standard_deviation = 0
                self.Q_Table = self.initialize_table()
            if i != 0 and i % 20 == 0:
                success_rate = success_num / 20
                SUCCESS_RATE[i] = success_rate
                success_num = 0

            Q_State_Values, Max_Q_State_Values = q_table_max_value_dict(self.Q_Table)
            Q_Table_Dataframe, Q_Value_Diff, standard_deviation = calculate_standard_deviation(Q_Value_Per_Episodes, Q_State_Values, Q_Value_Per_Episodes_Max, Q_Value_Diff,i)
            if 0 < standard_deviation < 1e-3:
                convergence_list.append(i)
                convergence_list_unique = np.unique(convergence_list)
                min_episode = min(convergence_list_unique)

            mse_difference = max((np.sum(np.power(Q_Table_Dataframe, 2))) / (Q_Table_Dataframe.shape[0] * Q_Table_Dataframe.shape[1]))
            q_convergence_list.append(mse_difference)

            while True:
                # env.render()
                next_state, reward, done, success_flag, fail_flag = self.env.step(current_action)
                # if random.uniform(0, 1) < (0.05+((i+1)/NUM_EPISODES)):
                #     actions_values = self.Q_Table[next_state]
                #     next_action = actions_values.index(max(actions_values))
                #     print('e', 1-(i/NUM_EPISODES) )
                # else:
                #     next_action = np.random.choice(self.actions)
                next_action = self.epsilon_greedy(epsilon_schedule, i,next_state)
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

                    # Record average rewards
                    rewards += reward
                    Reward_List += [rewards / (i + 1)]
                    break

                current_state = next_state
                current_action = next_action

            print("episodes:{}".format(i))
        print('SELF Q TABLE', self.Q_Table)
        # print('Max_Q_State_Values', Max_Q_State_Values)

        overall_success_rate = round(success_count / (success_count + fail_count), 2)
        return self.Q_Table,fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List

    def generate_heatmap(self, METHOD_NAME):
        global ax
        final_route = env.final()
        print('FINAL ROUTE', final_route)
        final_route_values_list = list(final_route.values())
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
        ax.get_figure().savefig('results/{}/{}/heatmap.jpg'.format(METHOD_NAME, MAP_SIZE), dpi=300, bbox_inches='tight', pad_inches=0.2)

    def plot_success_rate_epsilon(self,SUCCESS_RATE, METHOD_NAME, METHOD_PATH, SMOOTH):
        plt.figure(dpi=300)
        plt.rc('font', family='Times New Roman', size=12)
        max_success_rate = max(list(SUCCESS_RATE.values()))
        success_rate_smooth = pd.Series(SUCCESS_RATE.values()).rolling(SMOOTH, min_periods=5).mean()
        l1 = plt.scatter(SUCCESS_RATE.keys(), SUCCESS_RATE.values(), s=0.8, c='green', alpha=1.0, label='success rate point')
        l2 = plt.plot(SUCCESS_RATE.keys(), success_rate_smooth, 'b', label='success rate line')
        l3 = plt.axhline(y=max(list(SUCCESS_RATE.values())), c="r", ls="--", lw=2, label='maximum success rate: %s' %max_success_rate)
        plt.legend(handles=[l1, l2, l3], labels=['success rate point', 'success rate line', 'maximum success rate: %s' %max_success_rate], loc='best')
        plt.title('Method:{}   Metrics:Success Rate'.format(METHOD_NAME))
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        # plt.axhline(y=max(list(SUCCESS_RATE.values())), c="r", ls="--", lw=2,
        #             label='maximum success rate: %s' % max(list(SUCCESS_RATE.values())))
        plt.savefig('results/{}/{}/success_rate_epsilon.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.2)

if __name__ == '__main__':
    # create a FrozenLake environment
    # env = GUI()
    METHOD_NAME = 'SARSA Rising Epsilon'
    METHOD_PATH = 'SARSA_Epsilon'
    SMOOTH = SMOOTH_SIZE
    epsilon_schedule = 'increase'
    env = Environment()
    # Create a SARSA agent
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    # Learning and updating
    q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List = SARSA.train(epsilon_schedule,num_epoch=NUM_EPISODES)
    # plot_final_results(METHOD_NAME, METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List)
    # SARSA.generate_heatmap(METHOD_PATH)
    SARSA.plot_success_rate_epsilon(SUCCESS_RATE, METHOD_NAME, METHOD_PATH, SMOOTH)
    # env.mainloop()