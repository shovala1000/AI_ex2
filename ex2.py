import math
from time import time
import random
import matplotlib.pyplot as plt
import pacman

id = ["206626681"]

""" Rules """
WALL = 99
PACMAN = 70
RED_DOT = 21
RED = 20
BLUE_DOT = 31
BLUE = 30
YELLOW_DOT = 41
YELLOW = 40
GREEN_DOT = 51
GREEN = 50
DOT = 11
EMPTY_SLOT = 10
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

""" Invalid state """
INVALID_STATE = "Error!!"

""" Points """
EAT_DOT = 1
WIN = 10
LOST = -10

""" hyperparameters """
BANDIT_ARMS = 4
BANDIT_TRAINING = True
EPSILON = 1
EPISODES = 2
ALPHA = 0.8
GAMMA = 0.95
ACTIONS = {LEFT: (-1, 0), DOWN: (0, 1), RIGHT: (1, 0), UP: (0, -1)}


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
            for action in actions_keys:
                q_table[((i, j), action)] = 0

    # for row in q_table:
    #     print(row)
    return q_table


def max_actions_and_values(arm_rewards):
    max_value = max(arm_rewards.values())
    max_actions = [action for action, value in arm_rewards.items() if value == max_value]
    if len(max_actions) == 1:
        max_reward_arm = max_actions[0]
    else:
        # randomly choose an arm from the best arms
        max_reward_arm = random.choice(max_actions)
    return max_reward_arm


def get_neighboring_states(state, q_table, N, M):
    neighboring_states = []
    for (dx, dy), action in q_table:
        new_i, new_j = state[0] + dx, state[1] + dy
        if 0 <= new_i < N and 0 <= new_j < M:
            neighboring_states.append((new_i, new_j))
    return neighboring_states


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
    ax1.set_yticks([0, 1])
    ax2 = ax1.twinx()
    ax2.plot(episode_steps, '+r', label='step')
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("step")
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title("Training Progress")
    plt.show()


def run_epsilon_greedy(arm_rewards, arms, is_training, epsilon, episode):
    # check if bandit is used for training
    if is_training:
        # -> start with the full exploration and slowly reduce it as this algorithm learns
        epsilon = math.exp(-0.01 * episode)
        # flip the coin (randomly select the number between 0 and 1)
    random_num = random.random()
    # select the arm
    if random_num > epsilon:
        # Exploit -> find arms with the highest rewards
        max_reward_arm = max_actions_and_values(arm_rewards)
    else:
        # Explore -> randomly choose an arm
        max_reward_arm = random.choice(arms)

    return max_reward_arm


def run_Q_learning(env, steps, q_table):
    # log the training start
    training_start = time()

    # store the training progress of this algorithm for each episode
    episode_rewards = []
    episode_steps = []

    # solve the environment over certain amount of episodes
    for episode in range(EPISODES):
        # reset the environment, rewards, and steps for the new episode
        s = env.reset()
        episode_reward = 0
        step = 0

        # find the solution over certain amount of attempts (steps in each episode)
        while step < steps:

            # select the action in the current state by running the multiarmed bandit
            a = run_epsilon_greedy(q_table[s, :], BANDIT_ARMS, BANDIT_TRAINING, EPSILON, episode)

            # enter the environment and get the experience from it by performing there an action
            # -> get the observation (new state), reward, done (success/failure), and information

            # (observation, reward, done, info)
            reward = env.update_board(a)
            observation = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
            actions_and_values = {action: value for (state, action), value in q_table.items() if state == observation}
            max_action_q_s_tag = max_actions_and_values(actions_and_values)
            # update the Q-value for the current state and action
            # -> calculate this Q-value using its previous value & the experience from the environment
            q_table[s, a] = q_table[s, a] + ALPHA * (reward + GAMMA * max_action_q_s_tag - q_table[s, a])

            # add the reward to others during this episode
            episode_reward += reward

            # change the state to the observed state for the next iteration
            s = observation

            # check if the environment has been exited
            if env.done:
                # -> store the collected rewards & number of steps in this episode
                episode_rewards.append(episode_reward)
                episode_steps.append(step)
                # -> quit the episode
                break

            # continue looping
            step += 1

    # log the training end
    training_end = time()
    print_training_progress(episode_rewards, training_start, training_end, episode_steps)


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
        self.bandit_arms = BANDIT_ARMS
        self.is_training = BANDIT_TRAINING
        self.epsilon = EPSILON
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
        for row in self.q_table:
            print(row)

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
        return max_actions_and_values(actions_and_values)
