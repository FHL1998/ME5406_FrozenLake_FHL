import numpy as np

from GUI import GUI
from utils.Parameters import *
from Environment import Environment
from Monte_Carlo.Monte_Carlo import Monte_Carlo
from Q_Learning.Q_learning import Q_Learning
from SARSA.SARSA import SARSA
from utils.Utils import *

if __name__ == '__main__':
    np.random.seed(1)
    METHOD_NAME = ['Monte_Carlo', 'SARSA', 'Q_Learning']
    METHOD_PATH = ['Monte_Carlo', 'SARSA', 'Q_Learning']
    # env = Environment()
    env = GUI()  # uncomment this if you want to use Tkinter GUI
    SMOOTH = SMOOTH_SIZE

    ''' Implement the FV Monte Carlo, SARSA and Q Learning independently by creating three agents followed 
    by the three methods, the results will be stored separately in results/method name/map size '''
    Monte_Carlo = Monte_Carlo(env, epsilon=EPSILON, gamma=GAMMA)
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    Q_Learning = Q_Learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    ''' Run the main function of each methods and visualize the performance of each method'''
    # This is the 1st method: First Meet Monte Carlo Method
    q_table_1, fail_1, success_1, optimal_1, Episode_Time_1, Q_Value_Diff_1, min_episode_1, q_convergence_list_1, \
    steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Monte_Carlo.run()
    plot_final_results(METHOD_NAME[0], METHOD_PATH[0], SMOOTH, fail_1, success_1, optimal_1, Episode_Time_1,
                       Q_Value_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1,
                       overall_success_rate_1, Reward_List_1)
    # Monte_Carlo.generate_heatmap(METHOD_NAME[0])

    # This is the 2nd method: SARSA
    q_table_2, fail_2, success_2, optimal_2, Episode_Time_2, Q_Value_Diff_2, min_episode_2, q_convergence_list_2, \
    steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(NUM_EPISODES)
    plot_final_results(METHOD_NAME[1], METHOD_PATH[1], SMOOTH, fail_2, success_2, optimal_2, Episode_Time_2,
                       Q_Value_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2,
                       overall_success_rate_2, Reward_List_2)
    # SARSA.generate_heatmap(METHOD_NAME[1])

    # This is the 3rd method: Q Learning
    q_table_3, fail_3, success_3, optimal_3, Episode_Time_3, Q_Value_Diff_3, min_episode_3, q_convergence_list_3, \
    steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_Learning.train(NUM_EPISODES)
    plot_final_results(METHOD_NAME[2], METHOD_PATH[2], SMOOTH, fail_3, success_3, optimal_3, Episode_Time,
                       Q_Value_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3,
                       overall_success_rate_3, Reward_List_3)
    # Q_Learning.generate_heatmap(METHOD_NAME)
    env.mainllop()
