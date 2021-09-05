import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from Environment import Environment
from utils.Parameters import *
from utils.Utils import plot_diff_values, plot_q_convergence, plot_average_rewards, plot_success_rate
from SARSA import SARSA
random.seed(0)


if __name__ == "__main__":
    env = Environment()  # Create an environment
    SMOOTH = SMOOTH_SIZE
    METHOD_NAME = 'SARSA'
    METHOD_PATH = 'SARSA'
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)  # Create a q learning agent
    if TASK == TUNING_TASK_LIST[0]:
        label_learning_rate = ['learning rate:0.01', 'learning rate:0.05', 'learning rate:0.1', 'learning rate:0.3', 'learning rate:0.5']
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


        Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3, Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
        min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
        Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3, q_convergence_list_4, q_convergence_list_5]
        Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
        Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
        overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3, overall_success_rate_4, overall_success_rate_5]

        plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
        plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
        plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
        plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List, label_learning_rate)

    elif TASK == TUNING_TASK_LIST[1]:
        label_learning_rate = ['discount rate:0.2', 'discount rate:0.4', 'discount rate:0.6', 'discount rate:0.8', 'discount rate:1.0']
        SARSA.gamma = 0.2
        _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(NUM_EPISODES)
        SARSA.gamma = 0.4
        _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(NUM_EPISODES)
        SARSA.gamma = 0.6
        _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = SARSA.train(NUM_EPISODES)
        SARSA.gamma = 0.8
        _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = SARSA.train(NUM_EPISODES)
        SARSA.gamma = 1.0
        _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = SARSA.train(NUM_EPISODES)

        Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3, Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
        min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
        Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3, q_convergence_list_4, q_convergence_list_5]
        Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
        Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
        overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3, overall_success_rate_4, overall_success_rate_5]

        plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
        plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
        plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
        plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List, label_learning_rate)
    elif TASK == TUNING_TASK_LIST[2]:
        label_learning_rate = ['greedy policy:0.1', 'greedy policy:0.3', 'greedy policy:0.5', 'greedy policy:0.6', 'greedy policy:1.0(exploration)']
        SARSA.epsilon = 0.1
        _, _, _, _, Episode_Time_1, Q_Value_Per_Episodes_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(NUM_EPISODES)
        SARSA.epsilon = 0.3
        _, _, _, _, Episode_Time_2, Q_Value_Per_Episodes_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA.train(NUM_EPISODES)
        SARSA.epsilon = 0.5
        _, _, _, _, Episode_Time_3, Q_Value_Per_Episodes_Diff_3, min_episode_3, q_convergence_list_3, steps_3, SUCCESS_RATE_3, overall_success_rate_3, Reward_List_3 = SARSA.train(NUM_EPISODES)
        SARSA.epsilon = 0.6
        _, _, _, _, Episode_Time_4, Q_Value_Per_Episodes_Diff_4, min_episode_4, q_convergence_list_4, steps_4, SUCCESS_RATE_4, overall_success_rate_4, Reward_List_4 = SARSA.train(NUM_EPISODES)
        SARSA.epsilon = 1.0
        _, _, _, _, Episode_Time_5, Q_Value_Per_Episodes_Diff_5, min_episode_5, q_convergence_list_5, steps_5, SUCCESS_RATE_5, overall_success_rate_5, Reward_List_5 = SARSA.train(NUM_EPISODES)

        Q_Value_Diff_List = [Q_Value_Per_Episodes_Diff_1, Q_Value_Per_Episodes_Diff_2, Q_Value_Per_Episodes_Diff_3, Q_Value_Per_Episodes_Diff_4, Q_Value_Per_Episodes_Diff_5]
        min_episode_List = [min_episode_1, min_episode_2, min_episode_3, min_episode_4, min_episode_5]
        Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2, q_convergence_list_3, q_convergence_list_4, q_convergence_list_5]
        Reward_List = [Reward_List_1, Reward_List_2, Reward_List_3, Reward_List_4, Reward_List_5]
        Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2, SUCCESS_RATE_3, SUCCESS_RATE_4, SUCCESS_RATE_5]
        overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2, overall_success_rate_3, overall_success_rate_4, overall_success_rate_5]

        plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label_learning_rate)
        plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label_learning_rate)
        plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label_learning_rate)
        plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_List, label_learning_rate)