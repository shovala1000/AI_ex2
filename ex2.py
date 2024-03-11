import math
import time
import random

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

""" Global vars """
EPISODES = 1000
IS_TRAINING = True


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
        self.steps = steps
        self.alpha = 0.8
        self.gamma = 0.95
        self.epsilon = 1

        # Initialize game environment for visualization
        self.env = pacman.Game(steps, self.create_board(N, M, init_locations, init_pellets))
        self.states = []  # last state
        # self.actions = []  # last action

        self.actions = {LEFT: (-1, 0), DOWN: (0, 1), RIGHT: (1, 0), UP: (0, -1)}
        self.q_table = self.create_q_table(N, M)  # Initialize Q-table with zeros

    def create_board(self, N, M, init_locations, init_pellets):
        """
        Create a new board based on the initial locations and pellets.

        Returns:
        - A new board representing the environment
        """
        board = [[0] * M for _ in range(N)]
        for key, value in init_locations.items():
            if value:
                board[value[0]][value[1]] = key * 10
        for pellet in init_pellets:
            board[pellet[0]][pellet[1]] = 1
        # Print board
        for row in board:
            print(str(row) + "\n")
        return board

    def create_q_table(self, N, M):
        """
        Initialize Q-table and other necessary variables.
        """
        q_table = {}
        for i in range(N):
            for j in range(M):
                for action in self.actions.keys():
                    q_table[((i, j), action)] = 0
        return q_table

    def choose_next_move(self, locations, pellets):
        """
        Choose next action for Pacman given the current state of the board.
        """
        episode_rewards = []
        episode_steps = []

        for episode in range(EPISODES):
            self.env.reset()
            s = self.env.board  # init state - todo: maybe in our case it should be the pacman location
            episode_reward = 0
            step = 0
            while step < self.steps:
                a = self.run_epsilon_greedy(self.q_table[s, :], self.actions.keys(), True, self.epsilon, episode)

                # todo: continue the training part and then do the play

    def find_max_action(self, q_table, state):
        max_action = None
        max_q_value = float('-inf')

        for action in self.actions.keys():
            q_value = q_table.get((state, action), 0)
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action

        return max_action

    def run_epsilon_greedy(self, arm_rewards, arms, is_training, epsilon, episode):
        if is_training:
            epsilon = math.exp(-0.01 * episode)
        random_num = random.random()
        if random_num > epsilon:
            max_reward_arm = self.find_max_action(arm_rewards, (4, 0))
        else:
            max_reward_arm = random.choice(arms)
        return max_reward_arm
