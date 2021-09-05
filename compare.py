import numpy as np
from utils.Parameters import *
from Environment import Environment
from Monte_Carlo.Monte_Carlo import Monte_Carlo
from Q_Learning.Q_learning import Q_Learning
from SARSA.SARSA import SARSA
from utils.Utils import *

if __name__ == '__main__':
    np.random.seed(1)
    env = Environment()
    SMOOTH = SMOOTH_SIZE
    ''' Compare the performance of FV Monte Carlo, SARSA and Q Learning by creating three agents followed 
    by the three methods, the results will be stored in results/Compare/map size '''
    Monte_carlo = Monte_Carlo(env, epsilon=EPSILON, gamma=GAMMA)
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    Q_learning = Q_Learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    label_compare_algorithm = ['Monte_carlo', 'SARSA', 'Q_learning']

    # This is the 1st method: First Meet Monte Carlo Method
    _, _, _, _, Episode_Time_1, Q_Value_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Monte_carlo.run()

    # This is the 2nd method: SARSA
    _, _, _, _, Episode_Time_2, Q_Value_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(NUM_EPISODES)

    # This is the 3rd method: Q Learning
    _, _, _, _, Episode_Time_3, Q_Value_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_learning.train(NUM_EPISODES)

    # in order to compare the performance of the methods, the list is used to store the results achieved by different
    # methods, thus the results can be plotted in one figure for comparison
    Q_Value_Diff_List = [Q_Value_Diff_1, Q_Value_Diff_2, Q_Value_Diff_3]
    min_episode_List = [min_episode_1, min_episode_2, min_episode_3]
    Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3]
    Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3]
    Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3]
    overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3]

    # plot the relative figures to show the performance achieved by different methods such as convergence, success rate
    # and reward
    compare_diff_values(Q_Value_Diff_List, min_episode_List, label_compare_algorithm)
    compare_q_convergence(Q_Convergence_List, label_compare_algorithm)
    compare_success_rate(SMOOTH, Success_Rate_List, overall_success_rate_List, label_compare_algorithm)
    compare_average_rewards(Reward_List, label_compare_algorithm)
