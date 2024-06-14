import numpy as np
from Board import Board
from Direction import Direction
import gym
from gym import spaces

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.board = Board()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=int)

    def reset(self):
        self.board = Board()
        return self.board.board

    def step(self, action):
        direction = Direction(action + 1)
        self.board.move(direction)
        
        reward = -1
        done = False
        
        if np.any(self.board.board == 2048):
            reward = 10000
            done = True
        elif self.board.is_game_over():
            done = True

        return self.board.board, reward, done, {}

    def render(self):
        print(self.board.board)

    def close(self):
        pass
