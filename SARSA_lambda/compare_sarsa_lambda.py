from Environment import Environment

from SARSA.SARSA import SARSA
from SARSA_lambda.SARSA_Lambda import SARSA_Lambda

from utils.Utils import *


def compare_diff_values( Q_Value_Diff_List, min_episode_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(Q_Value_Diff_List[0].keys(), Q_Value_Diff_List[0].values(), 'r', label=label[0], linewidth=1)
    plt.plot(Q_Value_Diff_List[1].keys(), Q_Value_Diff_List[1].values(), 'g', label=label[1], linewidth=1)
    plt.title('Metrics:Standard Deviation')
    plt.xlabel('Episode')
    plt.ylabel('Difference')
    plt.axvline(x=min_episode_List[0], c="r", ls="--", lw=2, alpha=1, label='Convergence Episode: %s' % min_episode_List[0])
    plt.axvline(x=min_episode_List[1], c="g", ls="--", lw=2, alpha=0.8, label='Convergence Episode: %s' % min_episode_List[1])
    plt.legend(loc='best')
    plt.savefig('results/Improve/diff_value.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_q_convergence(SMOOTH, Q_Convergence_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Q_Convergence_List[0])), Q_Convergence_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Q_Convergence_List[1])), Q_Convergence_List[1], 'g', label=label[1], linewidth=1)
    plt.title('Metrics:MSE')
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.savefig('results/Improve/mse.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_success_rate(SMOOTH, Success_Rate_List, overall_success_rate_list, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    y_av_0 = pd.Series(list(Success_Rate_List[0].values())).rolling(SMOOTH, min_periods=5).mean()
    y_av_1 = pd.Series(list(Success_Rate_List[1].values())).rolling(SMOOTH, min_periods=5).mean()
    plt.plot(Success_Rate_List[0].keys(), y_av_0, 'r', label=label[0], linewidth=1)
    plt.plot(Success_Rate_List[1].keys(), y_av_1, 'g', label=label[1], linewidth=1)
    plt.title('Metrics:Success Rate(Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.axhline(y=overall_success_rate_list[0], c="r", ls="-.", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[0])
    plt.axhline(y=overall_success_rate_list[1], c="g", ls="--", lw=2,
                label='overall success rate: %s' % overall_success_rate_list[1])
    plt.legend(loc='best')
    plt.savefig('results/Improve/success_rate.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.show()


def compare_average_rewards(Reward_List, label):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure(dpi=300)
    plt.plot(np.arange(len(Reward_List[0])), Reward_List[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Reward_List[1])), Reward_List[1], 'g', label=label[1], linewidth=1)
    plt.title('Metrics:Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.legend(loc='best')
    plt.savefig('results/Improve/average_award.jpg', bbox_inches='tight',
                pad_inches=0.2)
    plt.show()




if __name__ == '__main__':
    np.random.seed(1)
    env = Environment()
    SMOOTH = SMOOTH_SIZE
    # Create three agents corresponding to three algorithms
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    SARSA_Lambda = SARSA_Lambda(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, lambda_=LAMBDA)

    label_compare_sarsa = ['SARSA', 'SARSA_Lambda']
    _, _, _, _, Episode_Time_1, Q_Value_Diff_1, min_episode_1, q_convergence_list_1, steps_1, SUCCESS_RATE_1, overall_success_rate_1, Reward_List_1 = SARSA.train(NUM_EPISODES)

    _, _, _, _, Episode_Time_2, Q_Value_Diff_2, min_episode_2, q_convergence_list_2, steps_2, SUCCESS_RATE_2, overall_success_rate_2, Reward_List_2 = SARSA_Lambda.update()
    Q_Value_Diff_List = [Q_Value_Diff_1, Q_Value_Diff_2]
    min_episode_List = [min_episode_1, min_episode_2]
    Q_Convergence_List = [q_convergence_list_1, q_convergence_list_2]
    Reward_List = [Reward_List_1, Reward_List_2]
    Success_Rate_List = [SUCCESS_RATE_1, SUCCESS_RATE_2]
    overall_success_rate_List = [overall_success_rate_1, overall_success_rate_2]

    compare_diff_values(Q_Value_Diff_List, min_episode_List, label_compare_sarsa)
    compare_q_convergence(SMOOTH, Q_Convergence_List, label_compare_sarsa)
    compare_success_rate(SMOOTH, Success_Rate_List, overall_success_rate_List, label_compare_sarsa)
    compare_average_rewards(Reward_List, label_compare_sarsa)