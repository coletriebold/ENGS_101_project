from __future__ import annotations

import copy
from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from cyberrl_eval import AgentConfig, Observation, Transition


GAMMA = 0.98
N_STEP = 3
LR = 3e-4
BATCH = 64
CAPACITY = 10_000
ALPHA = 0.6
BETA = 0.4
TARGET_EVERY = 200
EXPLORE_FRACTION = 0.65
FINAL_EPSILON = 0.05
EMA_DECAY = 0.999
EMA_START = 0.6
AVERAGE_WEIGHTS = {"red": True, "blue": False}


def dense_size(inputs, actions, hidden):
    return (inputs + actions + 1) * hidden + (hidden + 1) * hidden + (hidden + 1) + (hidden + 1) * actions


class DenseQ(nn.Module):
    def __init__(self, inputs, actions, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs + actions, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, actions)

    def forward(self, x, mask):
        h = self.body(torch.cat((x, mask.float()), -1))
        a = self.advantage(h)
        return self.value(h) + a - (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)


def infer_layouts(x, actions):
    layouts = []
    for hosts in range(1, (actions - 1) // 4 + 1):
        other = actions - 4 * hosts
        rest = len(x) - 1 - hosts * hosts
        if rest <= 0 or rest % hosts:
            continue
        per_host = rest // hosts
        if other > 1:
            roles = per_host - other - 12
            options = [(other, roles, 3, 3, 4, 2)] if 2 <= roles <= hosts + 1 else []
        else:
            options = [(s, r, 3, 4, 3, per_host - s - r - 10)
                       for s in range(1, min(hosts, per_host) + 1)
                       for r in range(2, min(hosts + 1, per_host) + 1)
                       if per_host - s - r - 10 >= 1]
        for sizes in options:
            offset, ok = 0, True
            for size in sizes:
                block = x[offset:offset + hosts * size].reshape(hosts, size)
                if not np.all((block == 0) | (block == 1)) or not np.all(block.sum(1) == 1):
                    ok = False
                    break
                offset += hosts * size
            if ok:
                layouts.append((hosts, other, *sizes))
    return layouts


class GraphQ(nn.Module):
    def __init__(self, obs_dim, actions, hidden=64):
        super().__init__()
        self.obs_dim, self.actions = obs_dim, actions
        self.register_buffer("layout", torch.zeros(8, dtype=torch.long))
        self.categories = nn.ModuleList([nn.Embedding(128, 8) for _ in range(6)])
        self.encode = nn.Sequential(nn.Linear(57, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.host_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.action_ids = nn.Embedding(actions, 8)
        self.prefix_head = nn.Sequential(nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Linear(hidden, 1)
        width = min(64, hidden)
        while width > 1 and dense_size(obs_dim + actions, actions, width) >= 700_000:
            width -= 1
        self.fallback = DenseQ(obs_dim + actions, actions, width)

    def forward(self, x, mask):
        if self.layout[0].item() < 1:
            return self.fallback(x, mask)
        single = x.ndim == 1
        if single:
            x, mask = x[None], mask[None]
        counts, x = x[:, self.obs_dim:], x[:, :self.obs_dim]
        hosts, other, *sizes = self.layout.tolist()
        offset, inputs = 0, []
        for embedding, size in zip(self.categories, sizes):
            inputs.append(embedding(x[:, offset:offset + hosts * size].reshape(-1, hosts, size).argmax(-1) % 128))
            offset += hosts * size
        inputs.append(mask[:, other:].reshape(-1, 4, hosts).transpose(1, 2).float())
        inputs.append(x[:, -1:].unsqueeze(1).expand(-1, hosts, -1))
        inputs.append(counts[:, other:].reshape(-1, 4, hosts).transpose(1, 2))
        h = self.encode(torch.cat(inputs, -1))
        adj = x[:, offset:-1].reshape(-1, hosts, hosts)
        neighbours = torch.bmm(adj, h) / adj.sum(-1, keepdim=True).clamp_min(1)
        g = h.mean(1)
        host_adv = self.host_head(torch.cat((h, neighbours, g[:, None, :].expand(-1, hosts, -1)), -1))
        host_adv = host_adv.transpose(1, 2).reshape(-1, 4 * hosts)
        ids = self.action_ids(torch.arange(other, device=x.device))[None].expand(x.shape[0], -1, -1)
        prefix_adv = self.prefix_head(torch.cat((g[:, None, :].expand(-1, other, -1), ids), -1)).squeeze(-1)
        adv = torch.cat((prefix_adv, host_adv), -1)
        q = self.value(g) + adv - (adv * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        return q[0] if single else q


class GraphDQNAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.device = torch.device(config.device)
        self.average = AVERAGE_WEIGHTS.get(config.role, False)
        obs_dim, actions = config.observation_dim, config.num_actions
        hidden = next(h for h in range(128, 0, -1) if dense_size(obs_dim, actions, h) <= 1_000_000)
        DenseQ(obs_dim, actions, hidden)  # unused: keeps the RNG stream (and initial weights) unchanged
        self.model = GraphQ(obs_dim, actions).to(self.device)
        self.target = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.rng = np.random.default_rng(config.seed)
        self.budget = max(1, config.total_steps)
        self.steps = 0
        self.pending = deque()
        self.capacity = min(CAPACITY, max(BATCH, 64 * 1024 * 1024 // (8 * obs_dim + 2 * actions + 32)))
        self.memory = []
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.cursor = 0
        self.counts = np.zeros(actions, dtype=np.float32)
        self.ema = None

    def reset(self, seed: int) -> None:
        self.pending.clear()
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.last = None

    def with_counts(self, observation):
        x = np.concatenate((observation.features, np.log1p(self.counts).astype(np.float32)))
        return SimpleNamespace(features=x, action_mask=observation.action_mask)

    def act(self, observation: Observation, training: bool) -> int:
        observation = self.last = self.with_counts(observation)
        if training and self.model.layout[0].item() == 0:
            layouts = infer_layouts(observation.features[:self.model.obs_dim], self.model.actions)
            if layouts:
                self.model.layout.copy_(torch.tensor(sorted(layouts)[0], dtype=torch.long, device=self.device))
            else:
                self.model.layout[0] = -1
            self.target.layout.copy_(self.model.layout)
        legal = np.flatnonzero(observation.action_mask)
        epsilon = 1.0 - (1.0 - FINAL_EPSILON) * min(1.0, self.steps / max(1.0, EXPLORE_FRACTION * self.budget))
        if training and self.rng.random() < epsilon:
            action = int(self.rng.choice(legal))
        else:
            with torch.no_grad():
                x = torch.as_tensor(observation.features, device=self.device)
                mask = torch.as_tensor(observation.action_mask, device=self.device)
                action = int(self.model(x, mask).masked_fill(~mask, -torch.inf).argmax().item())
        self.counts[action] += 1
        return action

    def store(self):
        first, last = self.pending[0], self.pending[-1]
        reward, discount = 0.0, 1.0
        for t in self.pending:
            reward += discount * t.reward
            discount *= GAMMA
            if t.done:
                discount = 0.0
                break
        item = (first.observation.features.copy(), first.observation.action_mask.copy(), first.action, reward,
                last.next_observation.features.copy(), last.next_observation.action_mask.copy(), discount)
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.cursor] = item
        self.priorities[self.cursor] = self.priorities[:len(self.memory)].max()
        self.cursor = (self.cursor + 1) % self.capacity
        self.pending.popleft()

    def sample(self):
        n = len(self.memory)
        p = self.priorities[:n] ** ALPHA
        p /= p.sum()
        idx = self.rng.choice(n, size=BATCH, p=p)
        w = (n * p[idx]) ** (-(BETA + (1 - BETA) * min(1.0, self.steps / self.budget)))
        w /= w.max()
        columns = []
        for column in zip(*[self.memory[i] for i in idx]):
            values = np.stack(column)
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32, copy=False)
            columns.append(torch.as_tensor(values, device=self.device))
        return idx, torch.as_tensor(w, dtype=torch.float32, device=self.device), columns

    def train_step(self):
        if len(self.memory) < BATCH:
            return
        idx, w, (x, mask, action, reward, next_x, next_mask, discount) = self.sample()
        q = self.model(x, mask).gather(1, action.long()[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.model(next_x, next_mask).masked_fill(~next_mask, -torch.inf).argmax(1, keepdim=True)
            target = reward.float() + discount.float() * self.target(next_x, next_mask).gather(1, best).squeeze(1)
        loss = (F.smooth_l1_loss(q, target, reduction="none") * w).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.priorities[idx] = (q - target).detach().abs().cpu().numpy() + 0.01

    def update_ema(self):
        if self.steps < EMA_START * self.budget:
            return
        with torch.no_grad():
            state = self.model.state_dict()
            if self.ema is None:
                self.ema = {k: v.detach().clone() for k, v in state.items()}
            else:
                for k, v in state.items():
                    if v.is_floating_point():
                        self.ema[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                    else:
                        self.ema[k].copy_(v)
        if self.steps >= self.budget:
            self.model.load_state_dict(self.ema)

    def learn(self, transition: Transition) -> None:
        self.pending.append(SimpleNamespace(observation=self.last, action=transition.action,
                                            reward=transition.reward, done=transition.done,
                                            next_observation=self.with_counts(transition.next_observation)))
        self.steps += 1
        if self.average:
            for group in self.optimizer.param_groups:
                group["lr"] = LR * (1.0 - 0.9 * min(1.0, self.steps / self.budget))
        if len(self.pending) >= N_STEP:
            self.store()
        if transition.done:
            while self.pending:
                self.store()
        self.train_step()
        if self.steps % TARGET_EVERY == 0:
            self.target.load_state_dict(self.model.state_dict())
        if self.average:
            self.update_ema()


def create_agent(config: AgentConfig) -> GraphDQNAgent:
    return GraphDQNAgent(config)
