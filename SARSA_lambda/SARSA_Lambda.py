import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from env import final_states

# backward eligibility traces
from Environment import Environment
from utils.Parameters import *
from utils.Utils import calculate_standard_deviation_Q_Learning, calculate_time, plot_final_results

random.seed(0)


class SARSA_Lambda:
    def __init__(self, env, learning_rate, gamma, epsilon, lambda_):
        self.env = env
        self.actions = list(range(ACTIONS_NUMBER))  # List of actions [0, 1, 2, 3]
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Value of discounted value
        self.epsilon = epsilon

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # backward view, eligibility trace.
        self.lambda_ = lambda_
        self.eligibility_trace = self.q_table.copy()

    def check_state_validation(self, state):
        """If the observation state is not in the q table, add a new row for the state to create a full q table ."""
        if state not in self.q_table.index:

            # In each state, initialize the Q value of each action to 0
            q_table_new_row = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(q_table_new_row)
            # need to update eligibility_trace
            self.eligibility_trace = self.eligibility_trace.append(q_table_new_row)

    def epsilon_greedy_policy(self, current_state):
        """Usage of epsilon greedy policy to balance exploration and exploitation.

        Args:
            current_state (str): index coordinate of states

        Returns:
            action(int): if the generalized random value > epsilon, execute optimal action,
                         if the generalized random value < epsilon, choose action randomly.
        """
        self.check_state_validation(current_state)
        # action selection
        if np.random.random() > self.epsilon:

            # gain the 4 Q value at current state corresponds to 4 actions
            state_action = self.q_table.loc[current_state, :]

            # Randomly find the position with the largest column value in a row,
            # and perform the action(the largest column value may be multiple, randomly selected)
            max_q_value_at_current_state = np.max(state_action)
            action = np.random.choice(state_action[state_action == max_q_value_at_current_state].index)
        else:
            action = np.random.choice(self.actions)  # Non-greedy mode randomly selects action
        return action

    def learn(self, state, action, reward, next_state, next_action):
        """The update manner of Q Learning to update action value at each state based on the pair(st,at,rt,st+1,at+1)
        Specifically, the update formulas are:
        Q(s,a) = Q(s,a) + alpha * TD error * E(s,a).
        E(s,a) <- gamma * lambda * E(s,a).

        Args:
            state (str): The observation state of the agent.
            action (int): The action selected based on the epsilon-greedy policy
            reward (int): The reward gained by taking the action
            next_state (str): The agent's observation state after implement the action
            next_action (int): The action selected based on the epsilon-greedy policy at next state
        """

        # Checking if the next step exists in the Q-table
        self.check_state_validation(next_state)
        q_predict = self.q_table.loc[state, action]

        # Calculate the q target value(estimated target value) according to update rules
        q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]  # next state is not terminal
        TD_error = q_target - q_predict

        # increase trace amount for visited state-action pair,
        # For the experienced state-action, we add 1 to prove that it is an indispensable part of the reward.
        self.eligibility_trace.loc[state, action] += 1

        # Updating action value in Q-table based on SARSA(lambda)
        # Sarsa (Lambda) has an additional eligibility_trace table, so there is a record of the exploration trajectory,
        # and this trajectory has a positive or negative impact on the value of the Q table.
        self.q_table += self.learning_rate * TD_error * self.eligibility_trace

        # The value of eligibility_trace decays with time.
        # the farther the step is from obtaining reward, the more "indispensable" it is.
        self.eligibility_trace *= self.gamma * self.lambda_

    def train(self, num_episode):
        global Q_Value_Per_Episodes, Q_Value_Per_Episodes_Max, Q_Value_Diff, convergence_list, \
            q_convergence_list, standard_deviation, min_episode
        Q_Value_Per_Episodes = {}  # The q value at certain state
        Q_Value_Diff = {}  # The difference between the q value at certain state and the average q value
        Q_Value_Per_Episodes_Max = {}  # The optimal action value at each state in the episodes
        convergence_list = []  # the number episodes that convergent
        q_convergence_list = []  # the list that store the mean square error value
        min_episode = 0  # the minimum episode number where start to convergent
        standard_deviation = 0  # the standard deviation of action value at certain state

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
            # initial observation state
            step = 0  # Initialize step count

            # initialize the observation state
            observation_state = self.env.reset()

            # action selection based on epsilon greedy policy
            action = self.epsilon_greedy_policy(str(observation_state))

            # initial all zero eligibility trace
            self.eligibility_trace *= 0

            if i != 0 and i % 20 == 0:
                success_rate = success_num / 20
                SUCCESS_RATE[i] = success_rate
                success_num = 0
            Q_Value_Diff, standard_deviation = calculate_standard_deviation_Q_Learning(Q_Value_Per_Episodes,
                                                                                       self.q_table,
                                                                                       Q_Value_Per_Episodes_Max,
                                                                                       Q_Value_Diff, i)
            if 0 < standard_deviation < 1e-3:
                convergence_list.append(i)
                convergence_list_unique = np.unique(convergence_list)
                min_episode = min(convergence_list_unique)

            mse_difference = max((np.sum(np.power(self.q_table, 2))) / (self.q_table.shape[0] * self.q_table.shape[1]))
            q_convergence_list.append(mse_difference)  # Append the MSE value to the q_convergence_list
            while True:
                observation_state_, reward, done, success_flag, fail_flag = self.env.step(action)

                # next action selection based on epsilon greedy policy at next observation state
                action_ = self.epsilon_greedy_policy(str(observation_state_))

                # update the action value of each state by SARSA(lambda) with transition (st,at,rt,st+1, at+1)
                self.learn(str(observation_state), action, reward, str(observation_state_), action_)

                # Swapping the observation states and the observation state before taking action
                observation_state = observation_state_

                # Swapping the next action and the current action
                action = action_
                step += 1
                # break while loop when end of this episode
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

                    rewards += reward  # Record total rewards to calculate average rewards
                    Reward_List += [rewards / (i + 1)]
                    break
            print('episode:{}'.format(i))

        # calculate the overall success rate based on the success number and fail number with two decimal places
        # specifically, it equals to success number/(success number+fail number)
        overall_success_rate = round(success_count / (success_count + fail_count), 2)

        return self.q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, \
               steps, SUCCESS_RATE, overall_success_rate, Reward_List


if __name__ == "__main__":
    # Create an environment
    METHOD_NAME = 'SARSA Lambda'
    METHOD_PATH = 'SARSA_Lambda'
    SMOOTH = SMOOTH_SIZE
    env = Environment()
    SARSA_Lambda = SARSA_Lambda(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, lambda_=LAMBDA)  # Create a q learning agent
    q_table, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List = SARSA_Lambda.train(num_episode=NUM_EPISODES)
    plot_final_results(METHOD_NAME, METHOD_PATH, SMOOTH, fail, success, optimal, Episode_Time, Q_Value_Diff, min_episode, q_convergence_list, steps, SUCCESS_RATE, overall_success_rate, Reward_List)