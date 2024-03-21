import math
import random
from time import time

import matplotlib.pyplot as plt
import pacman

id = ["206626681"]

""" Action keys """
RIGHT = "R"
LEFT = "L"
DOWN = "D"
UP = "U"

""" Locations keys """
KEY_PACMAN = 7
KEY_GREEN = 5
KEY_YELLOW = 4
KEY_BLUE = 3
KEY_RED = 2

""" hyperparameters """
EPISODES = 2000
ALPHA = 0.8
GAMMA = 0.95
ACTIONS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def create_board(N, M, init_locations, init_pellets):
    """
    Create a new board based on the initial locations and pellets.

    Returns:
    - A new board representing the environment
    """
    board = [[10] * M for _ in range(N)]
    for key, value in init_locations.items():
        if value:
            board[value[0]][value[1]] = key * 10
    for pellet in init_pellets:
        board[pellet[0]][pellet[1]] += 1
    return board


def create_q_table(N, M, actions_keys):
    """
    Initialize Q-table and other necessary variables.
    """
    q_table = {}
    for i in range(N):
        for j in range(M):
            actions_keys_copy = list(actions_keys)
            if i == 0:
                actions_keys_copy.remove(UP)
            elif i == N - 1:
                actions_keys_copy.remove(DOWN)
            if j == 0:
                actions_keys_copy.remove(LEFT)
            elif j == M - 1:
                actions_keys_copy.remove(RIGHT)
            for action in actions_keys_copy:
                q_table[((i, j), action)] = 0

    # for row in q_table:
    #     print(row)
    return q_table


def max_reward_action(arm_rewards):
    max_value = max(arm_rewards.values())
    max_actions = [action for action, value in arm_rewards.items() if value == max_value]
    if len(max_actions) == 1:
        max_reward_arm = max_actions[0]
    else:
        # randomly choose an arm from the best arms
        max_reward_arm = random.choice(max_actions)
    return max_reward_arm


def print_training_progress(episode_rewards, training_start, training_end, episode_steps):
    # show the success rate for solving the environment & elapsed training time
    success_rate = round((sum(episode_rewards) / EPISODES) * 100, 2)
    elapsed_training_time = int(training_end - training_start)
    print("\nThis environment has been solved", str(success_rate), "% of times over", str(EPISODES), "episodes within",
          str(elapsed_training_time), "seconds!")

    # plot the rewards and number of steps over all training episodes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(episode_rewards, '-g', label='reward')
    ax1.set_yticks([-10, 10])
    ax2 = ax1.twinx()
    ax2.plot(episode_steps, '+r', label='step')
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("step")
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title("Training Progress")
    plt.show()


def run_epsilon_greedy(arm_rewards, is_training, epsilon, episode):
    # check if bandit is used for training
    if is_training:
        # -> start with the full exploration and slowly reduce it as this algorithm learns
        epsilon = math.exp(-0.01 * episode)
        # flip the coin (randomly select the number between 0 and 1)
    random_num = random.random()
    # select the arm
    if random_num > epsilon:
        # Exploit -> find arms with the highest rewards
        max_reward_arm = max_reward_action(arm_rewards)
    else:
        # Explore -> randomly choose an arm
        max_reward_arm = random.choice(list(arm_rewards.keys()))

    return max_reward_arm


def run_Q_learning(env, steps, q_table):
    # log the training start
    # training_start = time()

    # store the training progress of this algorithm for each episode
    episode_rewards = []
    episode_steps = []

    # solve the environment over certain amount of episodes
    for episode in range(EPISODES):
        # reset the environment, rewards, and steps for the new episode
        env.reset()
        s = env.init_locations[KEY_PACMAN]
        episode_reward = 0
        step = 0

        # find the solution over certain amount of attempts (steps in each episode)
        while step < steps:
            # check if the environment has been exited
            if env.done:
                # -> store the collected rewards & number of steps in this episode
                episode_rewards.append(episode_reward)
                episode_steps.append(step)
                # -> quit the episode
                break

            arm_rewards = {action: value for (state, action), value in q_table.items() if state == s}

            # select the action in the current state by running the multiarmed bandit
            a = run_epsilon_greedy(arm_rewards, True, 1, episode)

            reward = env.update_board(ACTIONS[a])

            s_tag = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])

            actions_and_values = {action: value for (state, action), value in q_table.items() if state == s_tag}
            max_action_s_tag = max_reward_action(actions_and_values)

            # update the Q-value for the current state and action
            # -> calculate this Q-value using its previous value & the experience from the environment
            q_table[s, a] = q_table[s, a] + ALPHA * (
                    reward + GAMMA * actions_and_values[max_action_s_tag] - q_table[s, a])

            # add the reward to others during this episode
            episode_reward += reward

            # change the state to the observed state for the next iteration
            s = s_tag

            # continue looping
            step += 1

    # log the training end
    # training_end = time()
    # print_training_progress(episode_rewards, training_start, training_end, episode_steps)


def print_q_table(q_table):
    for x in range(5):
        actions = {UP: [-9, -9, -9, -9, -9],
                   DOWN: [-9, -9, -9, -9, -9],
                   LEFT: [-9, -9, -9, -9, -9],
                   RIGHT: [-9, -9, -9, -9, -9]}
        for y in range(5):
            state = (x, y)
            for action in actions.keys():
                if (state, action) in q_table:
                    actions[action][y] = q_table[(state, action)]
        for i in range(5):
            if actions[UP][i] >= 0:
                print(f"|UP = {actions[UP][i]:.3f}      ", end='')
            else:
                print(f"|UP = {actions[UP][i]:.3f}     ", end='')
        print()
        for i in range(5):
            if actions[DOWN][i] >= 0:
                print(f"|DOWN = {actions[DOWN][i]:.3f}    ", end='')
            else:
                print(f"|DOWN = {actions[DOWN][i]:.3f}   ", end='')
        print()
        for i in range(5):
            if actions[LEFT][i] >= 0:
                print(f"|LEFT = {actions[LEFT][i]:.3f}    ", end='')
            else:
                print(f"|LEFT = {actions[LEFT][i]:.3f}   ", end='')
        print()
        for i in range(5):
            if actions[RIGHT][i] >= 0:
                print(f"|RIGHT = {actions[RIGHT][i]:.3f}   ", end='')
            else:
                print(f"|RIGHT = {actions[RIGHT][i]:.3f}  ", end='')
        print(
            "\n---------------------------------------------------------------------------------------------------------")


class Controller:
    "This class is a controller for a Pacman game."

    def __init__(self, N, M, init_locations, init_pellets, steps):
        """Initialize controller for given game board and number of steps.
        This method MUST terminate within the specified timeout.
        N - board size along the coordinate y (number of rows)
        M - board size along the coordinate x (number of columns)
        init_locations - the locations of ghosts and Pacman in the initial state
        init_locations - the locations of pellets in the initial state
        steps - number of steps the controller will perform
        """
        # Hyperparameters
        self.episodes = EPISODES
        self.steps = steps
        self.is_training = True
        self.epsilon = 1
        self.alpha = ALPHA
        self.gamma = GAMMA

        # Initialize game environment for visualization
        self.env = pacman.Game(steps, create_board(N, M, init_locations, init_pellets))
        self.states = []  # all possible states
        self.actions = ACTIONS

        # Initialize Q-table with zeros
        self.q_table = create_q_table(N, M, self.actions.keys())

        # Training using Q-learning
        run_Q_learning(self.env, steps, self.q_table)
        print_q_table(self.q_table)
        # for key, value in self.q_table.items():
        #     print("Q[" + str(key) + "]: " + str(value))

        self.episodes = 10
        self.is_training = False
        # setting the agent for pure exploitation -> use only learned Q values from training
        self.epsilon = 0

    def choose_next_move(self, locations, pellets):
        """
        Choose next action for Pacman given the current state of the board.
        """
        s = locations[KEY_PACMAN]
        actions_and_values = {action: value for (state, action), value in self.q_table.items() if state == s}
        # a = run_epsilon_greedy(actions_and_values, self.is_training, self.epsilon, None)
        # return a
        return max_reward_action(actions_and_values)
        # todo: this: return max_reward_action(actions_and_values) or greedy
