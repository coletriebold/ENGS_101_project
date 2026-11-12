from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition


class SimpleAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.num_actions = config.num_actions
        self.model = nn.Linear(128, self.num_actions)

    def reset(self, seed: int) -> None:
        np.random.seed(seed)

    def act(self, observation: Observation, training: bool) -> int:
        state = tuple(observation.features)
        mask = np.asarray(observation.action_mask, dtype=bool)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        valid_actions = np.where(mask)[0]

        # Epsilon-greedy exploration during training
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Greedy action selection
        q_values = self.q_table[state].copy()
        q_values[~mask] = -np.inf
        return int(np.argmax(q_values))

    def learn(self, transition: Transition) -> None:
        state = tuple(transition.observation.features)
        next_state = tuple(transition.next_observation.features)
        action = transition.action
        reward = transition.reward
        done = transition.done

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        q_old = self.q_table[state][action]

        if done:
            q_target = reward
        else:
            mask = np.asarray(
                transition.next_observation.action_mask,
                dtype=bool
            )
            if mask.any():
                q_target = reward + self.gamma * np.max(
                    self.q_table[next_state][mask]
                )
            else:
                q_target = reward

        # Q-learning update
        self.q_table[state][action] = (
            q_old + self.alpha * (q_target - q_old)
        )


def create_agent(config: AgentConfig) -> SimpleAgent:
    return SimpleAgent(config)