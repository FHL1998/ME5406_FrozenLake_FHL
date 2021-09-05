import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils.Parameters import *

standard_deviation = 0
Episode_Time = {}


def calculate_standard_deviation_Q_Learning(Q_Value_Per_Episodes, Q_Table, Q_Value_Per_Episodes_Max,
                                            Q_Value_Per_Episodes_Diff, i):
    """ Calculate the difference change and the standard deviation of designated location to judge the convergence."""
    global standard_deviation
    # for map with size of 4, calculate start from 100th episode and every 10 episodes.
    if GRID_SIZE == 4:
        if i >= 200 and i % 10 == 0:
            Q_Value_Per_Episodes[i] = []
            for j in range(ACTIONS_NUMBER):
                Q_Value_Per_Episodes[i].append(Q_Table.loc[str(14), j])
    # for map with size of 10, calculate start from 1200th episode and every 20 episodes.
    elif GRID_SIZE == 10:
        if i >= 1000 and i % 20 == 0:
            Q_Value_Per_Episodes[i] = []
            for j in range(ACTIONS_NUMBER):
                if not Q_Table.loc[str(98), j] is None:
                    Q_Value_Per_Episodes[i].append(Q_Table.loc[str(98), j])

    for keys in Q_Value_Per_Episodes:
        values = np.max(Q_Value_Per_Episodes[keys])
        Q_Value_Per_Episodes_Max[keys] = values
        q_value_avg = np.mean(list(Q_Value_Per_Episodes_Max.values()))
        diff = abs(Q_Value_Per_Episodes_Max[keys] - q_value_avg)
        Q_Value_Per_Episodes_Diff[keys] = diff
        Q_Value_List = list(Q_Value_Per_Episodes_Diff.values())
        if len(Q_Value_List) > 20:
            standard_deviation = np.std(np.array(Q_Value_List[-20:]), ddof=1)

    return Q_Value_Per_Episodes_Diff, standard_deviation


def calculate_standard_deviation(Q_Value_Per_Episodes, Q_Table_Reindex, Q_Value_Per_Episodes_Max,
                                 Q_Value_Per_Episodes_Diff, i):
    """ This function is used particularly for Q-learning since the Q Table generated is in pandas.dataframe.
    Calculate the difference change and the standard deviation of designated location
    (the last state before reaching the frisbee) to judge the convergence."""
    global standard_deviation
    Q_Table_Dataframe = pd.DataFrame.from_dict(Q_Table_Reindex, orient='index', columns=['0', '1', '2', '3'])
    if GRID_SIZE == 4:
        if i >= 200 and i % 10 == 0:
            Q_Value_Per_Episodes[i] = []
            for j in range(ACTIONS_NUMBER):
                Q_Value_Per_Episodes[i].append(Q_Table_Dataframe.iloc[14, j])
    elif GRID_SIZE == 10:
        if i >= 1000 and i % 20 == 0:
            Q_Value_Per_Episodes[i] = []
            for j in range(ACTIONS_NUMBER):
                if not Q_Table_Dataframe.iloc[98, j] is None:
                    Q_Value_Per_Episodes[i].append(Q_Table_Dataframe.iloc[98, j])

    for keys in Q_Value_Per_Episodes:
        values = np.max(Q_Value_Per_Episodes[keys])
        Q_Value_Per_Episodes_Max[keys] = values
        q_value_avg = np.mean(list(Q_Value_Per_Episodes_Max.values()))
        diff = abs(Q_Value_Per_Episodes_Max[keys] - q_value_avg)
        Q_Value_Per_Episodes_Diff[keys] = diff
        Q_Value_List = list(Q_Value_Per_Episodes_Diff.values())
        if len(Q_Value_List) > 20:
            standard_deviation = np.std(np.array(Q_Value_List[-20:]), ddof=1)

    return Q_Table_Dataframe, Q_Value_Per_Episodes_Diff, standard_deviation


def q_table_to_array(full_Q_Table):
    """ This function is used for Q-learning since the Q Table generated is in pandas.dataframe and out of order.
        Convert the full q table to an array."""
    reindex_list = []
    full_Q_Table = full_Q_Table
    for i in range(0, GRID_SIZE ** 2):
        reindex_list.append(i)
    reindex_list = np.asarray(reindex_list, dtype=str)
    full_Q_Table_reindex = full_Q_Table.reindex(reindex_list)  # re_index the dataframe with 16 cells
    full_Q_Table_max = full_Q_Table_reindex.max(axis=1)  # 求dataframe 中每一行的最大值
    full_Q_Table_max_array = np.array(full_Q_Table_max).reshape(GRID_SIZE, GRID_SIZE)
    return full_Q_Table_max_array


def q_table_max_value_dict(Q_Table):
    """ This function is used to get the optimal action value at each state. As at each state, the full q table has 4
    values which corresponding to 4 actions."""
    Max_Q_State_Values = {}
    Q_State_Values = {}
    for keys in Q_Table.keys():
        action_q_value = Q_Table[keys]
        max_action_q_value = max(action_q_value)
        Q_State_Values[keys] = action_q_value
        Max_Q_State_Values[keys] = max_action_q_value
    return Q_State_Values, Max_Q_State_Values


def calculate_time(current_time, start_time, i):
    elapsed_time = current_time - start_time
    new_start_time = current_time
    start_time = new_start_time
    Episode_Time[i] = elapsed_time
    return start_time, Episode_Time


def plot_final_results(METHOD_NAME, METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff,
                       min_episode, q_convergence_list, steps, SUCCESS_RATE,
                       overall_success_rate, Reward_List):
    """ This function is used to plot the overall performance like steps length of
    success, fail, optimal, overall episode, time consuming, q value change, convergence episode, success rate and etc.
    during the training process."""
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    l1 = plt.scatter(success.keys(), success.values(), s=0.8, c='blue', alpha=1.0, label='Success')
    l2 = plt.vlines(optimal.keys(), 0, max(fail.values()), colors='green', alpha=0.1, label='Optimal')
    l3 = plt.scatter(fail.keys(), fail.values(), s=0.8, c='red', alpha=0.5, label='Fail')
    plt.legend(handles=[l1, l2, l3], labels=['Success', 'Optimal', 'Fail'], loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Step length')
    plt.title('Method:{}   Metrics:Steps'.format(METHOD_NAME))
    plt.savefig('results/{}/{}/steps.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.3)

    plt.figure()
    plt.plot(Episode_Time.keys(), Episode_Time.values(), 'b')
    plt.title('Method:{}   Metrics:Time'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.savefig('results/{}/{}/time.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.2)

    plt.figure(dpi=300)
    plt.plot(Q_Value_Diff.keys(), Q_Value_Diff.values(), 'b')
    plt.title('Method:{}   Metrics:Standard Deviation'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.axvline(x=min_episode, c="r", ls="--", lw=2, label='Convergence Episode: %s' % min_episode)
    plt.axvspan(min_episode, NUM_EPISODES, facecolor='g', alpha=0.3, **dict())
    plt.legend()
    plt.savefig('results/{}/{}/standard_deviation.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight',
                pad_inches=0.2)

    plt.figure(dpi=300)
    plt.plot(np.arange(len(q_convergence_list)), q_convergence_list, 'b')
    plt.title('Method:{}   Metrics:MSE of Q Values'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.savefig('results/{}/{}/mse_diff.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.2)

    plt.figure(dpi=300)
    success_rate_smooth = pd.Series(SUCCESS_RATE.values()).rolling(SMOOTH, min_periods=5).mean()
    plt.plot(SUCCESS_RATE.keys(), success_rate_smooth, 'b')
    plt.title('Method:{}   Metrics:Success Rate'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.axhline(y=overall_success_rate, c="r", ls="--", lw=2,
                label='overall success rate: %s' % overall_success_rate)
    plt.legend()
    plt.savefig('results/{}/{}/success_rate.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.2)

    plt.figure()
    plt.plot(np.arange(len(steps)), steps, 'b')
    plt.title('Method:{}   Metrics:Steps'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.figure(dpi=300)
    plt.boxplot([list(fail.values()), steps, list(success.values())], patch_artist=True, showmeans=True,
                widths=0.6, autorange=True, vert=False, showfliers=False,  # the outliers will be showed
                labels=['Fail', 'All Steps', 'Success'],
                boxprops={'color': 'black', 'facecolor': '#9999ff'},
                meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
                medianprops={'linestyle': '--', 'color': 'orange'})
    plt.title('Box Plot of Steps without Outliers')
    plt.xlabel('Steps During Single Episode')
    plt.savefig('results/{}/{}/success_fail_boxplot.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight',
                pad_inches=0.2)

    plt.figure()
    plt.plot(np.arange(len(Reward_List)), Reward_List, 'b')
    plt.title('Method:{}   Metrics:Average Rewards'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.savefig('results/{}/{}/average_awards.jpg'.format(METHOD_PATH, MAP_SIZE), bbox_inches='tight', pad_inches=0.2)
    plt.show()


def generate_heatmap(final_route, Q_Table, NAME):
    """ This function is used to plot the heatmap of the optimal action value.
    The first step is to gain the full q table. The second step is to gain the optimal action value at each state.
    The third step is to convert the q table into a square matrix whose size equals to grid size and then plot."""
    global ax
    plt.rc('font', family='Times New Roman', size=12)
    final_route.insert(0, (0, 0))
    full_Q_Table_max_array = q_table_to_array(Q_Table)
    sns.axes_style({'axes.edgecolor': 'black'})
    plt.rc('font', family='Times New Roman', size=12)
    if GRID_SIZE == 4:
        ax = sns.heatmap(full_Q_Table_max_array, mask=None, linewidths=.5, cmap="RdBu_r", annot=True, fmt='.2',
                         vmin=-0.6, vmax=1, annot_kws={'size': 13})
    elif GRID_SIZE == 10:
        ax = sns.heatmap(full_Q_Table_max_array, mask=MASK, linewidths=.3, cmap="RdBu_r", annot=False,
                         linecolor='black', vmin=-0.6, vmax=1, annot_kws={'size': 13})
    for rect in final_route:
        ax.add_patch(Rectangle(rect, 1, 1, fill=False, edgecolor='red', lw=1.5))
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=0, horizontalalignment='right')  # Set the coordinate arrangement of the y-axis
    plt.title('Maximum Action Value(%s)' % NAME)

    plt.show()


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    re = np.convolve(interval, window, 'same')
    return re


""" The following functions are used to demonstrate the different performance during the hyper-parameter tuning."""


def plot_diff_values(METHOD_NAME, METHOD_PATH, Q_Value_Diff_List, min_episode_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(Q_Value_Diff_List[0].keys(), Q_Value_Diff_List[0].values(), 'r', label=label[0], linewidth=1)
    plt.plot(Q_Value_Diff_List[1].keys(), Q_Value_Diff_List[1].values(), 'g', label=label[1], linewidth=1)
    plt.plot(Q_Value_Diff_List[2].keys(), Q_Value_Diff_List[2].values(), 'b', label=label[2], linewidth=1)
    plt.plot(Q_Value_Diff_List[3].keys(), Q_Value_Diff_List[3].values(), 'burlywood', label=label[3], linewidth=1)
    plt.plot(Q_Value_Diff_List[4].keys(), Q_Value_Diff_List[4].values(), 'c', label=label[4], linewidth=1)
    plt.title('Method:{}   Metrics:Standard Deviation'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.axvline(x=min_episode_List[0], c="r", ls="--", lw=2, alpha=1,
                label='Convergence Episode: %s' % min_episode_List[0])
    plt.axvline(x=min_episode_List[1], c="g", ls="--", lw=2, alpha=0.8,
                label='Convergence Episode: %s' % min_episode_List[1])
    plt.axvline(x=min_episode_List[2], c="b", ls="--", lw=2, alpha=0.6,
                label='Convergence Episode: %s' % min_episode_List[2])
    plt.axvline(x=min_episode_List[3], c="burlywood", ls="--", lw=2, alpha=0.5,
                label='Convergence Episode: %s' % min_episode_List[3])
    plt.axvline(x=min_episode_List[4], c="c", ls="--", lw=2, alpha=0.5,
                label='Convergence Episode: %s' % min_episode_List[4])
    plt.legend(loc='best')
    plt.savefig('results/{}/tuning/standard_deviation.jpg'.format(METHOD_PATH), bbox_inches='tight',
                pad_inches=0.2)
    plt.show()


def plot_q_convergence(METHOD_NAME, METHOD_PATH, Q_Convergence_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Q_Convergence_List[0])), Q_Convergence_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[1])), Q_Convergence_List[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[2])), Q_Convergence_List[2], 'b', label=label[2], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[3])), Q_Convergence_List[3], 'burlywood', label=label[3], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[4])), Q_Convergence_List[4], 'c', label=label[4], linewidth=1)
    plt.title('Episode via MSE')
    plt.title('Method:{}   Metrics:MSE of Q Values'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.savefig('results/{}/tuning/mse.jpg'.format(METHOD_PATH), bbox_inches='tight',
                pad_inches=0.2)
    plt.show()


def plot_average_rewards(METHOD_NAME, METHOD_PATH, Reward_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Reward_List[0])), Reward_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Reward_List[1])), Reward_List[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(Reward_List[2])), Reward_List[2], 'b', label=label[2], linewidth=1)
    plt.plot(np.arange(len(Reward_List[3])), Reward_List[3], 'burlywood', label=label[3], linewidth=1)
    plt.plot(np.arange(len(Reward_List[4])), Reward_List[4], 'c', label=label[4], linewidth=1)
    plt.title('Method:{}   Metrics:Average Rewards'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.legend(loc='best')
    plt.savefig('results/{}/tuning/average_award.jpg'.format(METHOD_PATH), bbox_inches='tight',
                pad_inches=0.2)
    plt.show()


def plot_success_rate(METHOD_NAME, METHOD_PATH, SMOOTH, Success_Rate_List, overall_success_rate_list, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    y_av_0 = pd.Series(list(Success_Rate_List[0].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_1 = pd.Series(list(Success_Rate_List[1].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_2 = pd.Series(list(Success_Rate_List[2].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_3 = pd.Series(list(Success_Rate_List[3].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_4 = pd.Series(list(Success_Rate_List[4].values())).rolling(SMOOTH, min_periods=5).mean()
    plt.plot(Success_Rate_List[0].keys(), y_av_0, 'r', label=label[0], linewidth=1)
    plt.plot(Success_Rate_List[1].keys(), y_av_1, 'g', label=label[1], linewidth=1)
    plt.plot(Success_Rate_List[2].keys(), y_av_2, 'b', label=label[2], linewidth=1)
    plt.plot(Success_Rate_List[3].keys(), y_av_3, 'burlywood', label=label[3], linewidth=1)
    plt.plot(Success_Rate_List[4].keys(), y_av_4, 'c', label=label[4], linewidth=1)
    plt.title('Method:{}   Metrics:Success Rate(moving average)'.format(METHOD_NAME))
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.axhline(y=overall_success_rate_list[0], c="r", ls="-.", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[0])
    plt.axhline(y=overall_success_rate_list[1], c="g", ls="--", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[1])
    plt.axhline(y=overall_success_rate_list[2], c="b", ls=":", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[2])
    plt.axhline(y=overall_success_rate_list[3], c="burlywood", ls=":", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[3])
    plt.axhline(y=overall_success_rate_list[4], c="c", ls=":", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[4])
    plt.legend(loc='best')
    plt.savefig('results/{}/tuning/success_rate.jpg'.format(METHOD_PATH), bbox_inches='tight',
                pad_inches=0.2)
    plt.show()


""" The following functions are used to compare the performance utilizing FV Monte Carlo,SARSA and Q Learning."""


def compare_diff_values(Q_Value_Diff_List, min_episode_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(Q_Value_Diff_List[0].keys(), Q_Value_Diff_List[0].values(), 'r', label=label[0], linewidth=1)
    plt.plot(Q_Value_Diff_List[1].keys(), Q_Value_Diff_List[1].values(), 'g', label=label[1], linewidth=1)
    plt.plot(Q_Value_Diff_List[2].keys(), Q_Value_Diff_List[2].values(), 'b', label=label[2], linewidth=1)
    plt.title('Metrics:Standard Deviation')
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.axvline(x=min_episode_List[0], c="r", ls="--", lw=2, alpha=1,
                label='Convergence Episode: %s' % min_episode_List[0])
    plt.axvline(x=min_episode_List[1], c="g", ls="--", lw=2, alpha=0.8,
                label='Convergence Episode: %s' % min_episode_List[1])
    plt.axvline(x=min_episode_List[2], c="b", ls="--", lw=2, alpha=0.6,
                label='Convergence Episode: %s' % min_episode_List[2])
    plt.legend(loc='best')
    plt.savefig('results/Compare/diff_value.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_q_convergence(Q_Convergence_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Q_Convergence_List[0])), Q_Convergence_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[1])), Q_Convergence_List[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[2])), Q_Convergence_List[2], 'b', label=label[2], linewidth=1)
    plt.title('Metrics:MSE')
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.savefig('results/Compare/mse.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_success_rate(SMOOTH, Success_Rate_List, overall_success_rate_list, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    y_av_0 = pd.Series(list(Success_Rate_List[0].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_1 = pd.Series(list(Success_Rate_List[1].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_2 = pd.Series(list(Success_Rate_List[2].values())).rolling(SMOOTH, min_periods=5).mean()
    plt.plot(Success_Rate_List[0].keys(), y_av_0, 'r', label=label[0], linewidth=1)
    plt.plot(Success_Rate_List[1].keys(), y_av_1, 'g', label=label[1], linewidth=1)
    plt.plot(Success_Rate_List[2].keys(), y_av_2, 'b', label=label[2], linewidth=1)
    plt.title('Metrics:Success Rate(Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.axhline(y=overall_success_rate_list[0], c="r", ls="-.", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[0])
    plt.axhline(y=overall_success_rate_list[1], c="g", ls="--", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[1])
    plt.axhline(y=overall_success_rate_list[2], c="b", ls=":", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[2])
    plt.legend(loc='best')
    plt.savefig('results/Compare/success_rate.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_average_rewards(Reward_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Reward_List[0])), Reward_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Reward_List[1])), Reward_List[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(Reward_List[2])), Reward_List[2], 'b', label=label[2], linewidth=1)
    plt.title('Metrics:Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.legend(loc='best')
    plt.savefig('results/Compare/average_award.jpg', bbox_inches='tight',
                pad_inches=0.2)
    plt.show()
