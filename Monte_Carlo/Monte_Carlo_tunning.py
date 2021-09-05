# File: GUI/map_10×10/SARSA/SARSA_run.py
# Description: Running algorithm
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2021 Fan Haolin
# github.com/FHL1998
#


# Importing classes
from env import Environment
from SARSA_algorithm import SarsaTable


def update():
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []

    for episode in range(200):
        # Initial obervation_state
        obervation_state = env.reset()

        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0

        # RL choose action based on obervation_state
        action = RL.choose_action(str(obervation_state))

        while True:
            # Refreshing environment
            env.render()

            # RL takes an action and get the next obervation_state and reward
            obervation_state_, reward, done = env.step(action)

            # RL choose action based on next obervation_state
            action_ = RL.choose_action(str(obervation_state_))

            # RL learns from the transition and calculating the cost
            cost += RL.learn(str(obervation_state), action, reward, str(obervation_state_), action_)

            # Swapping the obervation_states and actions - current and next
            obervation_state = obervation_state_
            action = action_

            # Calculating number of Steps in the current Episode
            i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or ice_hole
            if done:
                steps += [i]
                all_costs += [cost]
                break

    # Showing the final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs)


# Commands to be implemented after running this file
if __name__ == "__main__":
    # Calling for the environment
    env = Environment()
    # Calling for the main algorithm
    RL = SarsaTable(actions=list(range(env.NUM_ACTIONSctions)),
                    learning_rate=1,
                    reward_decay=0.9,
                    e_greedy=0.9)
    # Running the main loop with Episodes by calling the function update()
    env.after(100, update)  # Or just update()
    env.mainloop()
