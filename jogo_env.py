import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import sobrevivente as sv
import numpy as np

register(
    id='jogo-sobrevivencia',                               
    entry_point='jogo_env:SurvivorEnv',
)

class SurvivorEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 5}

    def __init__(self, grid_rows=5, grid_cols=5, render_mode=None, zombies_amount=2, supplies_amount=3, walls_amount=1):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        self.survivor = sv.Survivor(grid_rows=grid_rows, grid_cols=grid_cols, fps=self.metadata['render_fps'], zombies_amount=zombies_amount, supplies_amount=supplies_amount, walls_amount=walls_amount)
        self.action_space = spaces.Discrete(len(sv.SurvivorAction))

        self.observation_space = spaces.MultiDiscrete([grid_rows * grid_cols, self.survivor.supplies_amount + 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.survivor.reset(seed=seed)
        
        info = {}

        return self._get_obs(), info

  # não é possível fazer apenas a recompensa de 10 nos presentes, pois ele não consegue se achar para chegar ao final.
    def step(self, action):
        gridTile = self.survivor.perform_action(sv.SurvivorAction(action))

        reference_reward=-1
        finish=False
        if (gridTile == sv.GridTile.DOOR.value):
            reference_reward+=-10
            if (self.survivor.supplies_collected == self.survivor.supplies_amount):
                reference_reward += 100
            finish=True
        elif (gridTile == sv.GridTile.ZOMBIE.value):
           reference_reward+=-100
           finish=True
        elif (gridTile == sv.GridTile.SUPPLY.value):
           reference_reward+=10

        return self._get_obs(), reference_reward, finish, False, {}

    def render(self):
        self.survivor.render()

    def _get_obs(self):
        obs = np.zeros(2, dtype=np.int32)

        y, x = self.survivor.survivor_pos
        pos_val = y * self.grid_rows + x

        obs[0] = pos_val
        obs[1] = self.survivor.supplies_collected

        return obs
