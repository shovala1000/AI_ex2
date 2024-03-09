import time
import random

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
       
    def choose_next_move(self, locations, pellets):
        "Choose next action for Pacman given the current state of the board."
    
