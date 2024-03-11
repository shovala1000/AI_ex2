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
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.numTraining = 100
        self.episodesSoFar = 0  # Count the number of games we have played
        self.score = 0  # current score
        self.lastState = []  # last state
        self.lastAction = []  # last action

        self.steps = steps

        self.actions = {LEFT: (-1, 0), DOWN: (0, 1), RIGHT: (1, 0), UP: (0, -1)}
        self.state = init_locations[KEY_PACMAN]
        self.action = None  # Set initial action
        self.Q = {}  # Initialize Q-table with zeros
        self.init_q(N, M)

        # Initialize game environment for visualization
        self.game = pacman.Game(steps, self.create_board(N, M, init_locations, init_pellets))

    def create_board(self, N, M, init_locations, init_pellets):
        """
        Create a new board based on the initial locations and pellets.

        Returns:
        - A new board representing the environment
        """
        board = [[0] * M for _ in range(N)]
        for key, value in init_locations.items():
            board[value[0]][value[1]] = key * 10
        for pellet in init_pellets:
            board[pellet[0]][pellet[1]] = 1
        # Print board
        for row in board:
            print(str(row) + "\n")
        return board

    def init_q(self, N, M):
        """
        Initialize Q-table and other necessary variables.
        """
        for i in range(N):
            for j in range(M):
                for action in self.actions.keys():
                    self.Q[((i, j), action)] = 0

    def choose_next_move(self, locations, pellets):
        """
        Choose next action for Pacman given the current state of the board.
        """
        state = locations[KEY_PACMAN]
        possible_actions = list(self.actions.keys())

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        # Exploit: choose action with the highest Q-value for the current state
        max_q_value = float('-inf')
        best_action = None
        for action in possible_actions:
            q_value = self.Q.get((state, action), 0)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        return best_action



    def update_q(self, state, action, reward, q_max):
        """
        update Q value
        """
        q = self.Q.get(state, action)
        self.Q[(state, action)] = q + self.alpha * (reward + self.gamma * q_max - q)

    def play_game(self):
        """
        Execute the game using the Q-learning algorithm.
        """
        for _ in range(self.steps):
            # Choose next action
            move = self.choose_next_move(self.game.locations.copy(), self.game.pellets.copy())

            # Update state based on action
            reward = self.game.update_board(self.actions[move])
            new_state = self.game.locations[KEY_PACMAN]

            # Update Q-value using Bellman equation
            old_q_value = self.Q.get((self.state, self.action), 0)
            max_future_q_value = max([self.Q.get((new_state, a), 0) for a in self.actions.keys()])
            new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
            self.Q[(self.state, self.action)] = new_q_value

            # Update state and action for next iteration
            self.state = new_state
            self.action = move



