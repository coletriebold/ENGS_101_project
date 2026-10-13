"""A minimal trainable agent. Students may replace this implementation.

The submitted model may contain at most 1,000,000 registered parameters per
role. The local runner checks the same limit as the course server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np #added numpy
from torch import nn
import copy


if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition



class SimpleAgent:
    """One-layer, one-step policy-gradient example."""

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

    def reset(self, seed: int) -> None:
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
        logits = self.model(features).masked_fill(~mask, -torch.inf)

        if not training:
            return int(logits.argmax().item()) #returns the highest score action. keep this because we want greedy

        q_values = self.model(features).masked_fill(~mask, -torch.inf)
        if training and np.random.random() < 0.1:
            valid_actions = torch.where(mask)[0]
            return int(valid_actions[torch.randint(len(valid_actions), (1,), device=self.device)].item())
        return int(q_values.argmax().item())

    def learn(self, transition: Transition) -> None:
        state = torch.as_tensor(transition.observation.features, dtype=torch.float32, device=self.device) #creates state using transition input
        action = transition.action
        reward = torch.tensor(transition.reward, dtype=torch.float32, device=self.device) #both action and reward pulled straight from transition element
        next_state = torch.tensor(transition.next_observation.features, dtype=torch.float32, device = self.device)
        next_action_mask = torch.tensor(transition.next_observation.action_mask, dtype=torch.bool, device = self.device)
        done = transition.done
        gamma = 0.99

        q_values = self.model(state) #equal to forward pass through model at the current state. has values for every action
        q_predicted = q_values[action] #but this is the q-value of interest
        
        with torch.no_grad():
            if done:
                q_target = reward
            else:
                next_q_values = self.target_model(next_state)
                next_q_values[~next_action_mask] = float("-inf")
                q_target = reward + gamma*next_q_values.max()
        #line below essentially uses the torch loss functionality to implement difference between old q and new q
        loss = torch.nn.functional.smooth_l1_loss(q_predicted, q_target) #uses smooth_l1_loss because it is more resistance to high value errors
        ## in pseudocode, I want to calculate my q(s) for every s I visit, and then update the policy after each episode. repeat for episode
        #loss = -self._log_probability * reward #defines your loss function that you are minimizing via gradient descent
        #updates model. these stay the same
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        #update target every 100 steps
        if self.training_steps % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self._log_probability = None




def create_agent(config: AgentConfig) -> SimpleAgent:
    return SimpleAgent(config)
