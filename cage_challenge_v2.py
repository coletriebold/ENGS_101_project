"""CAGE-inspired V1: PPO + adaptive decoy priority + episodic memory.

Designed for the ENGS 101 cyberrl_eval SimpleAgent contract:
    create_agent(config), reset(seed), act(observation, training), learn(transition)

This is an adaptation, NOT a reproduction of CardiffUni's winning agent.
It uses a masked actor-critic PPO policy and a small deterministic Blue-only
heuristic that ranks decoy targets using mission value, observed action-mask
changes, deployment history, and threat-response opportunities. The environment
exposes one decoy action per host, unlike CAGE 2's multiple decoy types.

Action mapping assumed from the course scenario:
    Blue: 0 sleep; 1..9 analyse; 10..18 decoy; 19..27 remove; 28..36 restore.
    Red:  0 sleep; 1..3 discover; 4..12 scan; 13..21 exploit;
          22..30 escalate; 31..39 impact.

Only torch and the Python standard library are required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import random
import torch
from torch import nn
from torch.distributions import Categorical

if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, features):
        hidden = self.shared(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class SimpleAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.device = torch.device(config.device)
        self.num_actions = config.num_actions
        self.model = ActorCritic(config.observation_dim, config.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.ppo_epochs = 4
        self.rollout_length = 128
        self.minibatch_size = 32

        # Course-specific heuristic: Blue has 37 actions, decoys are 10..18.
        self.is_blue = self.num_actions == 37
        # Host order follows the scenario's host insertion order. Update these
        # priorities if the scenario topology changes. Server assets receive
        # higher weights based on mission costs in base.yaml.
        self.host_priority = [0.2, 0.3, 0.3, 0.6, 0.5, 0.5, 1.4, 1.3, 1.6]
        self.max_early_decoys = 3
        self.early_decoy_window = 12
        self.deployed_decoys = set()
        self.host_decoy_counts = [0] * 9
        self.host_activity = [0.0] * 9
        self.previous_mask = None
        self.episode_steps = 0
        self.rollout = []
        self.pending = None

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)
        random.seed(seed)
        self.deployed_decoys.clear()
        self.host_decoy_counts = [0] * 9
        self.host_activity = [0.0] * 9
        self.previous_mask = None
        self.episode_steps = 0
        self.pending = None
        # Do not discard rollout: PPO can learn from transitions across episodes.

    def _tensor_observation(self, observation):
        features = torch.as_tensor(observation.features, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=self.device)
        if not bool(mask.any()):
            raise ValueError('Observation has no valid actions')
        return features, mask

    def _distribution(self, features, mask):
        logits, value = self.model(features)
        logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits), value

    def _update_memory(self, mask):
        """Track observable changes in host-specific Blue action availability.

        This is a *proxy* for host activity, not a verified compromise signal.
        We deliberately avoid guessing undocumented feature-vector offsets.
        """
        if not self.is_blue:
            return
        current = mask.detach().cpu().tolist()
        if self.previous_mask is not None:
            for host in range(9):
                # Changes to analyse/remove/restore availability can be useful
                # indicators that something changed on a particular host.
                action_ids = (1 + host, 19 + host, 28 + host)
                changes = sum(current[a] != self.previous_mask[a] for a in action_ids)
                self.host_activity[host] = 0.85 * self.host_activity[host] + changes
        self.previous_mask = current

    def _adaptive_decoy(self, mask):
        """Choose a high-priority, not-yet-deployed valid decoy, or defer to PPO."""
        if not self.is_blue or self.episode_steps >= self.early_decoy_window:
            return None
        if sum(self.host_decoy_counts) >= self.max_early_decoys:
            return None

        # If the environment signals an actionable remove on a host whose
        # availability just changed, allow PPO to handle the potential threat.
        for host in range(9):
            if self.host_activity[host] > 0.9 and bool(mask[19 + host]):
                return None

        candidates = []
        for host in range(9):
            action = 10 + host
            if not bool(mask[action]) or self.host_decoy_counts[host] > 0:
                continue
            score = self.host_priority[host] + 0.5 * self.host_activity[host]
            candidates.append((score, -host, action))
        return max(candidates)[2] if candidates else None

    def act(self, observation: Observation, training: bool) -> int:
        features, mask = self._tensor_observation(observation)
        with torch.no_grad():
            dist, value = self._distribution(features, mask)
            self._update_memory(mask)
            forced_action = self._adaptive_decoy(mask)
            if forced_action is not None:
                action = forced_action
            elif training:
                action = int(dist.sample().item())
            else:
                action = int(dist.probs.argmax().item())

            if training:
                # PPO cannot train from a forced action as if it were sampled
                # from the actor; forced transitions still contribute to GAE.
                self.pending = {
                    'state': features.detach().clone(),
                    'mask': mask.detach().clone(),
                    'action': action,
                    'old_log_prob': float(dist.log_prob(torch.tensor(action, device=self.device)).item()),
                    'value': float(value.item()),
                    'forced': forced_action is not None,
                }
            else:
                self.pending = None

        if self.is_blue:
            if 10 <= action <= 18:
                self.deployed_decoys.add(action)
                self.host_decoy_counts[action - 10] += 1
            elif 28 <= action <= 36:
                # A restore may remove a decoy on the corresponding host.
                self.deployed_decoys.discard(action - 18)
                # A restore may clear deployed decoys. Avoid immediately
                # re-forcing them; the deployment history remains recorded.
        self.episode_steps += 1
        return action

    def learn(self, transition: Transition) -> None:
        if self.pending is None:
            return
        record = self.pending
        self.pending = None
        record['reward'] = float(transition.reward)
        record['done'] = bool(transition.done)
        self.rollout.append(record)

        if record['done']:
            self.deployed_decoys.clear()
            self.host_decoy_counts = [0] * 9
            self.host_activity = [0.0] * 9
            self.previous_mask = None
            self.episode_steps = 0

        if len(self.rollout) >= self.rollout_length or record['done']:
            if record['done']:
                last_value = 0.0
            else:
                next_state, _ = self._tensor_observation(transition.next_observation)
                with torch.no_grad():
                    _, last_value_tensor = self.model(next_state)
                    last_value = float(last_value_tensor.item())
            self._update_ppo(last_value)

    def _update_ppo(self, last_value: float) -> None:
        if not self.rollout:
            return
        # Generalized Advantage Estimation (GAE-Lambda).
        advantages = []
        gae = 0.0
        next_value = last_value
        for step in reversed(self.rollout):
            nonterminal = 0.0 if step['done'] else 1.0
            delta = step['reward'] + self.gamma * next_value * nonterminal - step['value']
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages.append(gae)
            next_value = step['value']
        advantages.reverse()

        states = torch.stack([x['state'] for x in self.rollout])
        masks = torch.stack([x['mask'] for x in self.rollout])
        actions = torch.tensor([x['action'] for x in self.rollout], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([x['old_log_prob'] for x in self.rollout], device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_values = torch.tensor([x['value'] for x in self.rollout], device=self.device)
        returns = advantages + old_values
        policy_eligible = torch.tensor([not x['forced'] for x in self.rollout], dtype=torch.bool, device=self.device)

        # Normalization is helpful when sufficient policy-selected actions exist.
        if int(policy_eligible.sum().item()) > 1:
            eligible_adv = advantages[policy_eligible]
            advantages = (advantages - eligible_adv.mean()) / (eligible_adv.std(unbiased=False) + 1e-8)

        n = len(self.rollout)
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                logits, values = self.model(states[idx])
                logits = logits.masked_fill(~masks[idx], -1e9)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                ratios = (new_log_probs - old_log_probs[idx]).exp()
                unclipped = ratios * advantages[idx]
                clipped = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                eligible = policy_eligible[idx]
                if bool(eligible.any()):
                    policy_loss = -torch.minimum(unclipped[eligible], clipped[eligible]).mean()
                    entropy = dist.entropy()[eligible].mean()
                else:
                    policy_loss = torch.zeros((), device=self.device)
                    entropy = torch.zeros((), device=self.device)
                value_loss = 0.5 * (values - returns[idx]).square().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
        self.rollout.clear()


def create_agent(config: AgentConfig) -> SimpleAgent:
    return SimpleAgent(config)
