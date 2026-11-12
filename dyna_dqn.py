from __future__ import annotations

from typing import TYPE_CHECKING

from collections import defaultdict
import torch
import numpy as np #added numpy
from torch import nn
import copy
import random
import torch.nn.functional as F


if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition



class SimpleAgent:
    """attempt at Dyna-DQN"""

    def __init__(self, config: AgentConfig) -> None:
        self.device = torch.device(config.device)
        self.model = nn.Sequential(
            nn.Linear(config.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_actions)).to(self.device) # this builds the neural network with 3 linear layers. 128 is arbitrary choice
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self._log_probability = None
        self.target_model = copy.deepcopy(self.model)
        self.training_steps = 0

        self.gamma = 0.99
        self.epsilon = 0.1
        self.planning_steps = 5 #modify this to see what happens
        self.target_update_interval = 200
        self.training_steps = 0

        # Key: (state bytes, action). Value: observed one-step outcomes.
        # Each outcome contains next state, reward, terminal flag, next mask.
        self.transition_model = defaultdict(list)
        self.seen_pairs = []
        self.max_outcomes_per_pair = 20
        self.rng = random.Random()

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)
        torch.manual_seed(seed)

    def act(self, observation: Observation, training: bool) -> int:
        #returns an action based on self, it knows when you are training
        features = torch.as_tensor(
            observation.features,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.as_tensor(
            observation.action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        valid_actions = torch.where(mask)[0]
        if valid_actions.numel() == 0:
            raise ValueError("No valid actions in observation")

        if training and self.rng.random() < self.epsilon:
            return int(valid_actions[self.rng.randrange(len(valid_actions))].item())

        with torch.no_grad():
            q_values = self.model(features).masked_fill(~mask, -torch.inf)
            return int(q_values.argmax().item())

    def _update_q(self, state, action, reward, next_state, done, next_mask):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.bool, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        predicted = self.model(state_t)[action]
        with torch.no_grad():
            if done or not bool(next_mask_t.any()):
                target = reward_t
            else:
                next_q = self.target_model(next_state_t).masked_fill(
                    ~next_mask_t, -torch.inf
                )
                target = reward_t + self.gamma * next_q.max()

        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def learn(self, transition: Transition) -> None:
        state = tuple(float(x) for x in transition.observation.features)
        next_state = tuple(float(x) for x in transition.next_observation.features)
        next_mask = tuple(bool(x) for x in transition.next_observation.action_mask)
        action = int(transition.action)
        reward = float(transition.reward)
        done = bool(transition.done)

        # 1. Learn from the actual environment transition.
        self._update_q(state, action, reward, next_state, done, next_mask)

        # 2. Update empirical one-step model.
        key = (state, action)
        if key not in self.transition_model:
            self.seen_pairs.append(key)
        outcomes = self.transition_model[key]
        outcome = (next_state, reward, done, next_mask)
        if len(outcomes) < self.max_outcomes_per_pair:
            outcomes.append(outcome)
        else:
            outcomes[self.rng.randrange(self.max_outcomes_per_pair)] = outcome

        # 3. Planning: sample observed (state, action) and model outcome.
        for _ in range(self.planning_steps):
            sampled_state, sampled_action = self.rng.choice(self.seen_pairs)
            sampled_next_state, sampled_reward, sampled_done, sampled_mask = (
                self.rng.choice(self.transition_model[(sampled_state, sampled_action)])
            )
            self._update_q(
                sampled_state, sampled_action, sampled_reward,
                sampled_next_state, sampled_done, sampled_mask,
            )




def create_agent(config: AgentConfig) -> SimpleAgent:
    return SimpleAgent(config)
