"""Tuned five-step prioritized dueling Double DQN.

Standalone course submission: copy this file to student_agent.py, then run:
    python run.py --steps 10000 --eval-episodes 3 --device cpu

The same architecture, learner and hyperparameters serve Red and Blue;
the runner trains separate weights using each role's supplied dimensions.
Shared settings: five-step returns, dueling Double DQN, prioritized replay,
128/128 ReLU layers, Adam lr=0.0001, gamma=0.99, batch=64, target=200,
epsilon 1.0->0.05 over the first 65% of training, and capacity=10000.
Learned inference state is registered in model for checkpoint serialization.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Shared across both roles. Optimized at the submission's 10,000-step budget.
PARAMETERS = {'width': 128,
 'lr': 0.0001,
 'gamma': 0.99,
 'n_step': 5,
 'batch': 64,
 'warmup': 64,
 'capacity': 10000,
 'alpha': 0.6,
 'beta': 0.4,
 'target': 200,
 'tau': 0.0,
 'decay': 0.65,
 'eps_end': 0.05,
 'dueling': True,
 'masks': True,
 'layernorm': False,
 'activation': 'relu',
 'quantiles': 1,
 'reward_scale': 1.0,
 'lr_decay': False,
 'updates': 1,
 'update_every': 1,
 'weight_decay': 0.0}


class QNetwork(nn.Module):
    """Dueling state/action values with the valid-action mask as an input."""

    def __init__(self, config, p, copies=1):
        super().__init__()
        obs, actions = (config.observation_dim, config.num_actions)
        self.actions, self.nq = (actions, p['quantiles'])
        self.masks, self.dueling = (p['masks'], p['dueling'])
        width = p['width']
        while True:
            input_dim = obs + (actions if self.masks else 0)
            output_dim = (actions + int(self.dueling)) * self.nq
            count = width * input_dim + width + width * width + width + (width + 1) * output_dim
            if p['layernorm']:
                count += 4 * width
            if copies * count <= 1000000:
                break
            width -= 1
            if width < 1:
                raise ValueError('Model dimensions exceed parameter budget')
        activation = nn.Tanh if p['activation'] == 'tanh' else nn.ReLU
        layers = [nn.Linear(input_dim, width)]
        if p['layernorm']:
            layers.append(nn.LayerNorm(width))
        layers += [activation(), nn.Linear(width, width)]
        if p['layernorm']:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        self.body = nn.Sequential(*layers)
        if self.dueling:
            self.value = nn.Linear(width, self.nq)
            self.advantage = nn.Linear(width, actions * self.nq)
        else:
            self.output = nn.Linear(width, actions * self.nq)

    def distribution(self, x, m):
        h = self.body(torch.cat((x, m.float()), -1) if self.masks else x)
        if self.dueling:
            a = self.advantage(h).reshape(*h.shape[:-1], self.actions, self.nq)
            mean = (a * m[..., None]).sum(-2, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1)[..., None]
            return self.value(h)[..., None, :] + a - mean
        return self.output(h).reshape(*h.shape[:-1], self.actions, self.nq)

    def forward(self, x, m):
        return self.distribution(x, m).mean(-1)

class Replay:
    """Bounded proportional prioritized replay with importance corrections."""

    def __init__(self, config, p):
        bytes_per = 8 * config.observation_dim + 2 * config.num_actions + 32
        self.capacity = min(p['capacity'], max(p['batch'], 64 * 1024 * 1024 // bytes_per))
        self.data = []
        self.priorities = np.ones(self.capacity, np.float32)
        self.cursor = 0

    def add(self, item):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.cursor] = item
        self.priorities[self.cursor] = self.priorities[:len(self.data)].max()
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, p, rng, progress, device):
        n = len(self.data)
        if p['alpha'] > 0:
            probs = self.priorities[:n] ** p['alpha']
            probs /= probs.sum()
            ids = rng.choice(n, size=p['batch'], p=probs)
            beta = p['beta'] + (1 - p['beta']) * progress
            weights = (n * probs[ids]) ** (-beta)
            weights /= weights.max()
        else:
            ids = rng.integers(n, size=p['batch'])
            weights = np.ones(p['batch'], np.float32)
        batch = [self.data[i] for i in ids]
        tensors = []
        for column in zip(*batch):
            values = np.stack(column)
            # Python rewards/returns become float64 in NumPy; MPS needs float32.
            # Convert before device transfer, preserving integer actions and masks.
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32, copy=False)
            tensors.append(torch.as_tensor(values, device=device))
        return (ids, torch.as_tensor(weights, dtype=torch.float32, device=device), tensors)

class DQN:
    """One shared five-step Double DQN learner for either role."""

    def __init__(self, config, p):
        self.p = p.copy()
        self.device = torch.device(config.device)
        self.model = QNetwork(config, p).to(self.device)
        self.target = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        self.rng = np.random.default_rng(config.seed)
        self.total_steps = max(1, config.total_steps)
        self.steps = 0
        self.pending = deque()
        self.replay = Replay(config, p)

    def reset(self, seed):
        self.pending.clear()
        self.rng = np.random.default_rng(seed)

    def act(self, o, training):
        valid = np.flatnonzero(o.action_mask)
        if not len(valid):
            raise ValueError('Observation has no valid action')
        epsilon = 1.0 - (1.0 - self.p['eps_end']) * min(1.0, self.steps / (self.p['decay'] * self.total_steps))
        if training and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            x = torch.as_tensor(o.features, device=self.device)
            m = torch.as_tensor(o.action_mask, device=self.device)
            return int(self.model(x, m).masked_fill(~m, -torch.inf).argmax().item())

    def _store(self):
        """Store an n-step return; terminal windows may contain fewer steps."""
        first = self.pending[0]
        reward, discount = (0.0, 1.0)
        for t in self.pending:
            reward += discount * t.reward * self.p['reward_scale']
            discount *= self.p['gamma']
            if t.done:
                discount = 0.0
                break
        last = self.pending[-1]
        self.replay.add((first.observation.features.copy(), first.observation.action_mask.copy(), first.action, reward, last.next_observation.features.copy(), last.next_observation.action_mask.copy(), discount))
        self.pending.popleft()

    def learn(self, t):
        self.steps += 1
        self.pending.append(t)
        if len(self.pending) >= self.p['n_step']:
            self._store()
        if t.done:
            while self.pending:
                self._store()
        if len(self.replay.data) < max(self.p['warmup'], self.p['batch']) or self.steps % self.p['update_every']:
            return
        progress = min(1.0, self.steps / self.total_steps)
        if self.p['lr_decay']:
            for group in self.optimizer.param_groups:
                group['lr'] = self.p['lr'] * (1.0 - 0.9 * progress)
        for _ in range(self.p['updates']):
            ids, w, (x, m, a, r, nx, nm, d) = self.replay.sample(self.p, self.rng, progress, self.device)
            if self.p['quantiles'] == 1:
                pred = self.model(x, m).gather(1, a.long()[:, None]).squeeze(1)
                with torch.no_grad():
                    best = self.model(nx, nm).masked_fill(~nm, -torch.inf).argmax(1, keepdim=True)
                    boot = self.target(nx, nm).gather(1, best).squeeze(1)
                    target = r.float() + d.float() * boot
                errors = F.smooth_l1_loss(pred, target, reduction='none')
                priority = (pred - target).detach().abs()
            else:
                nq = self.p['quantiles']
                pred = self.model.distribution(x, m)[torch.arange(len(a), device=self.device), a.long()]
                with torch.no_grad():
                    best = self.model(nx, nm).masked_fill(~nm, -torch.inf).argmax(1)
                    boot = self.target.distribution(nx, nm)[torch.arange(len(a), device=self.device), best]
                    target = r.float()[:, None] + d.float()[:, None] * boot
                delta = target[:, None, :] - pred[:, :, None]
                tau = (torch.arange(nq, device=self.device, dtype=torch.float32) + 0.5) / nq
                huber = F.smooth_l1_loss(pred[:, :, None].expand_as(delta), target[:, None, :].expand_as(delta), reduction='none')
                errors = ((tau[None, :, None] - (delta.detach() < 0).float()).abs() * huber).mean((1, 2))
                priority = (target.mean(1) - pred.mean(1)).detach().abs()
            loss = (errors * w).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.replay.priorities[ids] = priority.cpu().numpy() + 0.01 * self.p['reward_scale']
            if self.p['tau']:
                with torch.no_grad():
                    for target, online in zip(self.target.parameters(), self.model.parameters()):
                        target.lerp_(online, self.p['tau'])
        if not self.p['tau'] and self.steps % self.p['target'] == 0:
            self.target.load_state_dict(self.model.state_dict())


def create_agent(config):
    """Build the same selected learner and settings for either role."""
    return DQN(config, PARAMETERS)
