import math
import random
import pacman

id = ["206626681"]

""" Action keys """
RIGHT = "R"
LEFT = "L"
DOWN = "D"
UP = "U"

""" Locations keys """
KEY_PACMAN = 7

""" hyperparameters """
EPISODES = 2000
ALPHA = 0.8
GAMMA = 0.95
ACTIONS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def create_board(rows, cols, init_locations, init_pellets):
    """
    Create a new board based on the initial locations and pellets.

    Returns:
    - A new board representing the environment
    """
    board = [[10] * cols for _ in range(rows)]

    for key, value in init_locations.items():
        if value:
            board[value[0]][value[1]] = key * 10
    for pellet in init_pellets:
        board[pellet[0]][pellet[1]] += 1
    return board


def create_q_table(rows, cols, actions_keys):
    """
    Initialize Q-table all zero, only when to move is in the board
    ("smart" table - without illegal moves)
    """
    q_table = {}
    for i in range(rows):
        for j in range(cols):
            actions_keys_copy = list(actions_keys)
            if i == 0:
                actions_keys_copy.remove(UP)
            elif i == rows - 1:
                actions_keys_copy.remove(DOWN)
            if j == 0:
                actions_keys_copy.remove(LEFT)
            elif j == cols - 1:
                actions_keys_copy.remove(RIGHT)
            for action in actions_keys_copy:
                q_table[((i, j), action)] = 0

    return q_table


def max_reward_action(arm_rewards):
    """
    Return the arm with the max q_value. In tie choose randomly.
    """
    max_value = max(arm_rewards.values())
    max_actions = [action for action, value in arm_rewards.items() if value == max_value]
    if len(max_actions) == 1:
        max_reward_arm = max_actions[0]
    else:
        # randomly choose an arm from the best arms
        max_reward_arm = random.choice(max_actions)
    return max_reward_arm


def run_epsilon_greedy(arm_rewards, is_training, epsilon, episode):
    """
    epsilon greedy algorithm.
    """
    # check if bandit is used for training
    if is_training:
        # start with the full exploration and slowly reduce it as this algorithm learns
        epsilon = math.exp(-0.01 * episode)
        # flip the coin (randomly select the number between 0 and 1)
    random_num = random.random()
    # select the arm
    if random_num > epsilon:
        # Exploit - find arms with the highest rewards
        max_reward_arm = max_reward_action(arm_rewards)
    else:
        # Explore - randomly choose an arm
        max_reward_arm = random.choice(list(arm_rewards.keys()))

    return max_reward_arm


def run_Q_learning(env, steps, q_table):
    """
    Q-learning algorithm, using epsilon-greedy algorithm.
    """
    # solve the environment over certain amount of episodes
    for episode in range(EPISODES):
        # reset the environment, rewards, and steps for the new episode
        env.reset()
        s = env.init_locations[KEY_PACMAN]
        step = 0

        # find the solution over certain amount of attempts (steps in each episode)
        while step < steps:
            # check if the environment has been exited
            if env.done:
                # quit the episode
                break

            arm_rewards = {action: value for (state, action), value in q_table.items() if state == s}

            # select the action in the current state by running the multiarmed bandit
            a = run_epsilon_greedy(arm_rewards, True, 1, episode)

            reward = env.update_board(ACTIONS[a])

            s_tag = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])

            actions_and_values = {action: value for (state, action), value in q_table.items() if state == s_tag}
            max_action_s_tag = max_reward_action(actions_and_values)

            # update the Q-value for the current state and action
            q_table[s, a] = q_table[s, a] + ALPHA * (
                    reward + GAMMA * actions_and_values[max_action_s_tag] - q_table[s, a])

            # change the state to the observed state for the next iteration
            s = s_tag

            # continue looping
            step += 1


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
        self.env = pacman.Game(steps, create_board(M, N, init_locations, init_pellets))
        self.states = []  # all possible states
        self.actions = ACTIONS

        # Initialize Q-table with zeros
        self.q_table = create_q_table(M, N, self.actions.keys())

        # Training using Q-learning
        run_Q_learning(self.env, steps, self.q_table)

        # setting the agent for pure exploitation -> use only learned Q values from training
        self.episodes = 10
        self.is_training = False
        self.epsilon = 0

    def choose_next_move(self, locations, pellets):
        """
        Choose next action for Pacman given the current state of the board.
        """
        s = locations[KEY_PACMAN]
        actions_and_values = {action: value for (state, action), value in self.q_table.items() if state == s}
        return max_reward_action(actions_and_values)
