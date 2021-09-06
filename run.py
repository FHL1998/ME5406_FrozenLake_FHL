import argparse

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
    parser = argparse.ArgumentParser(description="Variables that need to be declared")

    # Add the argument for the OVERALL_TASK, the options are in the OVERALL_TASK_LIST
    # OVERALL_TASK_LIST = ['Run Three Methods', 'Compare Three Methods', 'Tuning Q Learning', 'Tuning SARSA']
    parser.add_argument('overall_task', type=str, default='Run Three Methods',
                        help='The overall task agent could perform.')

    # TUNING_TASK_LIST = ['None', 'Tuning Learning Rate', 'Tuning Discount Rate', 'Tuning Greedy Policy']
    # When overall_task is 'Run Three Methods' or 'Compare Three Methods' ,tuning_task should be 'None'
    parser.add_argument('tuning_task', type=str, default='Tuning Learning Rate',
                        help='The sub-task agent can perform when the overall task is Tuning.')
    args = parser.parse_args()
    env = Environment()
    # env = GUI()  # uncomment this if you want to use Tkinter GUI
    Monte_Carlo = Monte_Carlo(env, epsilon=EPSILON, gamma=GAMMA)
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    Q_Learning = Q_Learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    SMOOTH = SMOOTH_SIZE
    if args.overall_task == 'Run Three Methods':
        if args.tuning_task == 'None':
            METHOD_NAME = ['Monte_Carlo', 'SARSA', 'Q_Learning']
            METHOD_PATH = ['Monte_Carlo', 'SARSA', 'Q_Learning']

            ''' Implement the FV Monte Carlo, SARSA and Q Learning independently by creating three agents followed 
            by the three methods, the results will be stored separately in results/method name/map size '''

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
            # env.mainloop()

    elif args.overall_task == 'Compare Three Methods':
        if args.tuning_task == 'None':
            ''' Compare the performance of FV Monte Carlo, SARSA and Q Learning by creating three agents followed 
            by the three methods, the results will be stored in results/Compare/map size '''

            label_compare_algorithm = ['Monte_carlo', 'SARSA', 'Q_learning']

            # This is the 1st method: First Meet Monte Carlo Method
            _, _, _, _, Episode_Time_1, Q_Value_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Monte_Carlo.run()

            # This is the 2nd method: SARSA
            _, _, _, _, Episode_Time_2, Q_Value_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(
                NUM_EPISODES)

            # This is the 3rd method: Q Learning
            _, _, _, _, Episode_Time_3, Q_Value_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_Learning.train(
                NUM_EPISODES)

            # in order to compare the performance of the methods, the list is used to store the results achieved by
            # different methods, thus the results can be plotted in one figure for comparison
            Q_Value_Diff_List = [Q_Value_Diff_1, Q_Value_Diff_2, Q_Value_Diff_3]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3]

            # plot the relative figures to show the performance achieved by different methods such as
            # convergence, success rate and reward
            compare_diff_values(Q_Value_Diff_List, min_episode_List, label_compare_algorithm)
            compare_q_convergence(Q_Convergence_List, label_compare_algorithm)
            compare_success_rate(SMOOTH, Success_Rate_List, overall_success_rate_List, label_compare_algorithm)
            compare_average_rewards(Reward_List, label_compare_algorithm)

    elif args.overall_task == 'Tuning Q Learning':
        METHOD_NAME = 'Q Learning'
        METHOD_PATH = 'Q_Learning'

        # Tuning learning rate for Q Learning
        if args.tuning_task == 'Tuning Learning Rate':
            label_learning_rate = ['learning rate:0.01', 'learning rate:0.05', 'learning rate:0.1', 'learning rate:0.3',
                                   'learning rate:0.5']
            Q_Learning.learning_rate = 0.01
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.learning_rate = 0.05
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.learning_rate = 0.1
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.learning_rate = 0.3
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.learning_rate = 0.5
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = Q_Learning.train(
                NUM_EPISODES)

            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)

        # Tuning discount rate for Q Learning
        elif args.tuning_task == 'Tuning Discount Rate':
            label_learning_rate = ['discount rate:0.2', 'discount rate:0.4', 'discount rate:0.6', 'discount rate:0.8',
                                   'discount rate:1.0']
            Q_Learning.gamma = 0.2
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.gamma = 0.4
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.gamma = 0.6
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.gamma = 0.8
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.gamma = 1.0
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = Q_Learning.train(
                NUM_EPISODES)

            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)

        # Tuning epsilon greedy for Q Learning
        elif args.tuning_task == 'Tuning Greedy Policy':
            label_learning_rate = ['greedy policy:0.1', 'greedy policy:0.3', 'greedy policy:0.5', 'greedy policy:0.6',
                                   'greedy policy:1.0(exploration)']
            Q_Learning.epsilon = 0.1
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.epsilon = 0.3
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.epsilon = 0.5
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.epsilon = 0.6
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = Q_Learning.train(
                NUM_EPISODES)
            Q_Learning.epsilon = 1.0
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = Q_Learning.train(
                NUM_EPISODES)
            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)

    elif args.overall_task == 'Tuning SARSA':
        METHOD_NAME = 'SARSA'
        METHOD_PATH = 'SARSA'

        # Tuning learning rate for SARSA
        if args.tuning_task == 'Tuning Learning Rate':
            label_learning_rate = ['learning rate:0.01', 'learning rate:0.05', 'learning rate:0.1', 'learning rate:0.3',
                                   'learning rate:0.5']
            SARSA.learning_rate = 0.01
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(
                NUM_EPISODES)
            SARSA.learning_rate = 0.05
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(
                NUM_EPISODES)
            SARSA.learning_rate = 0.1
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = SARSA.train(
                NUM_EPISODES)
            SARSA.learning_rate = 0.3
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = SARSA.train(
                NUM_EPISODES)
            SARSA.learning_rate = 0.5
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = SARSA.train(
                NUM_EPISODES)

            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)

        # Tuning discount rate for SARSA
        elif args.tuning_task == 'Tuning Discount Rate':
            label_learning_rate = ['discount rate:0.2', 'discount rate:0.4', 'discount rate:0.6', 'discount rate:0.8',
                                   'discount rate:1.0']
            SARSA.gamma = 0.2
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(
                NUM_EPISODES)
            SARSA.gamma = 0.4
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(
                NUM_EPISODES)
            SARSA.gamma = 0.6
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = SARSA.train(
                NUM_EPISODES)
            SARSA.gamma = 0.8
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = SARSA.train(
                NUM_EPISODES)
            SARSA.gamma = 1.0
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = SARSA.train(
                NUM_EPISODES)

            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)

        # Tuning epsilon greedy for SARSA
        elif args.tuning_task == 'Tuning Greedy Policy':
            label_learning_rate = ['greedy policy:0.1', 'greedy policy:0.3', 'greedy policy:0.5', 'greedy policy:0.6',
                                   'greedy policy:1.0(exploration)']
            SARSA.epsilon = 0.1
            _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(
                NUM_EPISODES)
            SARSA.epsilon = 0.3
            _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(
                NUM_EPISODES)
            SARSA.epsilon = 0.5
            _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = SARSA.train(
                NUM_EPISODES)
            SARSA.epsilon = 0.6
            _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = SARSA.train(
                NUM_EPISODES)
            SARSA.epsilon = 1.0
            _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = SARSA.train(
                NUM_EPISODES)

            Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3,
                                 Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
            min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
            Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3,
                                  q_convergence_list_4, q_convergence_list_5]
            Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
            Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
            overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3,
                                         overall_success_rate_4, overall_success_rate_5]

            plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
            plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
            plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
            plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List,
                              label_learning_rate)
