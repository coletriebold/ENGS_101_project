"""Shared-host graph Double DQN for the two-stage cyber RL assignment.

The same learner, architecture and hyperparameters serve both roles. Host and
topology sizes are inferred from the supplied public observation/action schema;
there are no host-name lookups or map-specific action indices.

The graph network pools information across hosts so a useful update at one
host can improve decisions elsewhere. The DDQN/replay implementation below
is standalone. All learned weights and layout metadata live in model."""

from collections import deque
import copy
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HIDDEN_UNITS = 128
LEARNING_RATE = 0.0003
DISCOUNT_FACTOR = 0.98
RETURN_STEPS = 3
BATCH_SIZE = 64
WARMUP_TRANSITIONS = 64
REPLAY_CAPACITY = 10_000
PRIORITY_EXPONENT = 0.6
INITIAL_IMPORTANCE_CORRECTION = 0.4
TARGET_UPDATE_INTERVAL = 200
EXPLORATION_FRACTION = 0.65
FINAL_RANDOM_ACTION_PROBABILITY = 0.05
MODEL_PARAMETER_LIMIT = 1_000_000


def linear_layer_parameters(input_units, output_units):
    weights = input_units * output_units
    biases = output_units
    return weights + biases


def q_network_parameters(observation_size, action_count, hidden_units):
    input_units = observation_size + action_count
    first_hidden_layer = linear_layer_parameters(input_units, hidden_units)
    second_hidden_layer = linear_layer_parameters(hidden_units, hidden_units)
    state_value_head = linear_layer_parameters(hidden_units, 1)
    action_advantage_head = linear_layer_parameters(hidden_units, action_count)
    return (
        first_hidden_layer
        + second_hidden_layer
        + state_value_head
        + action_advantage_head
    )


def choose_hidden_units(config, parameter_budget):
    for hidden_units in range(HIDDEN_UNITS, 0, -1):
        parameter_count = q_network_parameters(
            config.observation_dim, config.num_actions, hidden_units
        )
        if parameter_count <= parameter_budget:
            return hidden_units
    raise ValueError('Observation/action dimensions exceed the model budget')


class QNetwork(nn.Module):


    def __init__(self, config, hidden_units):
        super().__init__()
        input_units = config.observation_dim + config.num_actions
        self.body = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )

        self.value = nn.Linear(hidden_units, 1)
        self.advantage = nn.Linear(hidden_units, config.num_actions)

    def forward(self, features, action_mask):
        network_input = torch.cat((features, action_mask.float()), dim=-1)
        hidden = self.body(network_input)
        state_value = self.value(hidden)
        advantages = self.advantage(hidden)


        legal_advantage_sum = (advantages * action_mask).sum(dim=-1, keepdim=True)
        legal_action_count = action_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        average_advantage = legal_advantage_sum / legal_action_count
        return state_value + advantages - average_advantage


class Experience(NamedTuple):
    features: np.ndarray
    action_mask: np.ndarray
    action: int
    accumulated_reward: float
    next_features: np.ndarray
    next_action_mask: np.ndarray
    future_discount: float
    episode_return: float = 0.0
    return_is_known: bool = False


class ReplayBuffer:
    def __init__(self, config):

        observation_bytes = 2 * config.observation_dim * 4
        action_mask_bytes = 2 * config.num_actions
        scalar_bytes = 32
        bytes_per_experience = observation_bytes + action_mask_bytes + scalar_bytes
        array_memory_budget = 64 * 1024 * 1024
        memory_limited_capacity = array_memory_budget // bytes_per_experience
        self.capacity = min(
            REPLAY_CAPACITY, max(BATCH_SIZE, memory_limited_capacity)
        )
        self.data = []
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.cursor = 0  # Next slot to fill; wrap around when the buffer is full.

    def add(self, experience):
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            self.data[self.cursor] = experience
        # Give a new experience the highest existing priority so it gets seen.
        self.priorities[self.cursor] = self.priorities[:len(self.data)].max()
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, random_generator, training_progress, device):
        experience_count = len(self.data)
        probabilities = self.priorities[:experience_count] ** PRIORITY_EXPONENT
        probabilities /= probabilities.sum()
        indices = random_generator.choice(
            experience_count, size=BATCH_SIZE, p=probabilities
        )

        # Sampling is deliberately biased toward high-priority experiences.
        # These weights correct that bias, increasingly as training progresses.
        correction_strength = INITIAL_IMPORTANCE_CORRECTION + (
            1 - INITIAL_IMPORTANCE_CORRECTION
        ) * training_progress
        importance_weights = (
            experience_count * probabilities[indices]
        ) ** (-correction_strength)
        importance_weights /= importance_weights.max()

        sampled_experiences = [self.data[index] for index in indices]
        batch_fields = {}
        for field_name in Experience._fields:
            values = np.stack([
                getattr(experience, field_name)
                for experience in sampled_experiences
            ])
            # MPS cannot accept float64. Cast before sending data to the device,
            # while preserving integer actions and Boolean legality masks.
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32, copy=False)
            batch_fields[field_name] = torch.as_tensor(values, device=device)

        weights = torch.as_tensor(
            importance_weights, dtype=torch.float32, device=device
        )
        return indices, weights, SimpleNamespace(**batch_fields)


class DoubleDQNAgent:
    """Five-step, prioritized, dueling Double DQN.

    learn() has two jobs: save the latest experience, then update the network
    from a randomly sampled batch of past experiences.
    """

    def __init__(self, config, parameter_budget=MODEL_PARAMETER_LIMIT):
        self.device = torch.device(config.device)
        hidden_units = choose_hidden_units(config, parameter_budget)
        self.model = QNetwork(config, hidden_units).to(self.device)
        # The target is a slowly refreshed copy used to calculate stable goals.
        self.target = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
        )
        self.random_generator = np.random.default_rng(config.seed)
        self.training_budget = max(1, config.total_steps)
        self.training_steps = 0
        self.pending_transitions = deque()
        self.replay = ReplayBuffer(config)

    def reset(self, seed):
        self.pending_transitions.clear()
        self.random_generator = np.random.default_rng(seed)

    def act(self, observation, training):
        legal_actions = np.flatnonzero(observation.action_mask)
        if len(legal_actions) == 0:
            raise ValueError('Observation has no valid action')

        # Explore heavily at first, then retain a 5% random-action probability.
        exploration_steps = EXPLORATION_FRACTION * self.training_budget
        exploration_progress = min(1.0, self.training_steps / exploration_steps)
        random_action_probability = 1.0 - (
            1.0 - FINAL_RANDOM_ACTION_PROBABILITY
        ) * exploration_progress
        if training and self.random_generator.random() < random_action_probability:
            return int(self.random_generator.choice(legal_actions))

        with torch.no_grad():
            features = torch.as_tensor(observation.features, device=self.device)
            action_mask = torch.as_tensor(observation.action_mask, device=self.device)
            action_values = self.model(features, action_mask)
            action_values = action_values.masked_fill(~action_mask, -torch.inf)
            return int(action_values.argmax().item())

    def _build_experience(self):
        """Combine up to five rewards: r0 + gamma*r1 + gamma^2*r2 + ..."""
        first = self.pending_transitions[0]
        last = self.pending_transitions[-1]
        accumulated_reward = 0.0
        future_discount = 1.0
        for transition in self.pending_transitions:
            accumulated_reward += future_discount * transition.reward
            future_discount *= DISCOUNT_FACTOR
            if transition.done:
                future_discount = 0.0  # No future value after an episode ends.
                break
        return Experience(
            features=first.observation.features.copy(),
            action_mask=first.observation.action_mask.copy(),
            action=first.action,
            accumulated_reward=accumulated_reward,
            next_features=last.next_observation.features.copy(),
            next_action_mask=last.next_observation.action_mask.copy(),
            future_discount=future_discount,
        )

    def _store_oldest_transition(self):
        self.replay.add(self._build_experience())
        self.pending_transitions.popleft()

    def _remember_transition(self, transition):
        self.pending_transitions.append(transition)
        if len(self.pending_transitions) >= RETURN_STEPS:
            self._store_oldest_transition()
        if transition.done:
            # Save the remaining short windows too, rather than dropping the
            # last four actions of every episode.
            while self.pending_transitions:
                self._store_oldest_transition()

    def _calculate_targets(self, batch):
        """Double DQN: the online model chooses; the target model evaluates."""
        with torch.no_grad():
            online_values = self.model(batch.next_features, batch.next_action_mask)
            online_values = online_values.masked_fill(
                ~batch.next_action_mask, -torch.inf
            )
            best_actions = online_values.argmax(dim=1, keepdim=True)
            target_values = self.target(batch.next_features, batch.next_action_mask)
            future_values = target_values.gather(1, best_actions).squeeze(1)
            return (
                batch.accumulated_reward.float()
                + batch.future_discount.float() * future_values
            )

    def _train_from_replay(self):
        if len(self.replay.data) < max(WARMUP_TRANSITIONS, BATCH_SIZE):
            return
        progress = min(1.0, self.training_steps / self.training_budget)
        indices, importance_weights, batch = self.replay.sample(
            self.random_generator, progress, self.device
        )
        all_action_values = self.model(batch.features, batch.action_mask)
        chosen_actions = batch.action.long()[:, None]
        predicted_values = all_action_values.gather(1, chosen_actions).squeeze(1)
        target_values = self._calculate_targets(batch)

        # Huber loss is less sensitive to large errors than squared error.
        errors = F.smooth_l1_loss(predicted_values, target_values, reduction='none')
        loss = (errors * importance_weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # Larger prediction errors mean a transition is worth revisiting.
        prediction_errors = (predicted_values - target_values).detach().abs()
        self.replay.priorities[indices] = prediction_errors.cpu().numpy() + 0.01
        if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
            self.target.load_state_dict(self.model.state_dict())

    def learn(self, transition):
        self.training_steps += 1
        self._remember_transition(transition)
        self._train_from_replay()


def infer_observation_layouts(features, actions):
    """Infer sizes from the public one-hot observation/action schema.
    Four host-action families exist on each side. Counts of hosts, subnets,
    roles and decoys are inferred, never fixed to the public example map.
    """
    observation_size = len(features)
    candidates = []
    for host_count in range(1, (actions - 1) // 4 + 1):
        non_host_actions = actions - 4 * host_count
        remaining = observation_size - 1 - host_count * host_count
        if remaining <= 0 or remaining % host_count:
            continue
        features_per_host = remaining // host_count
        if non_host_actions > 1:
            subnets = non_host_actions - 1
            role_categories = features_per_host - (subnets + 1) - 12
            if not 2 <= role_categories <= host_count + 1:
                continue
            possible_category_counts = [(subnets + 1, role_categories, 3, 3, 4, 2)]
        else:
            possible_category_counts = []
            for subnets in range(1, min(host_count, features_per_host) + 1):
                for role_categories in range(2, min(host_count + 1, features_per_host) + 1):
                    decoy_categories = features_per_host - subnets - role_categories - 10
                    if decoy_categories >= 1:
                        possible_category_counts.append((subnets, role_categories, 3, 4, 3, decoy_categories))
        for category_counts in possible_category_counts:
            feature_offset = 0
            valid = True
            for category_count in category_counts:
                one_hot_values = features[feature_offset:feature_offset + host_count * category_count].reshape(host_count, category_count)
                if not np.all((one_hot_values == 0) | (one_hot_values == 1)) or not np.all(one_hot_values.sum(1) == 1):
                    valid = False
                    break
                feature_offset += host_count * category_count
            if valid:
                candidates.append((host_count, non_host_actions, *category_counts))
    return candidates

class SharedHostQNetwork(nn.Module):
    """Share one learned decision rule across hosts, regardless of map size.

    Encode each host, average its neighbors' representations, and predict four
    action-family advantages. Global context predicts non-host actions.
    Actual host, subnet and action counts come from configuration/observations.
    An ordinary dense Q network handles unrecognized observation layouts.
    """

    def __init__(self, config, hidden_units=64):
        super().__init__()
        self.actions = config.num_actions
        self.observation_size = config.observation_dim
        self.register_buffer('layout', torch.zeros(8, dtype=torch.long))
        self.categories = nn.ModuleList([nn.Embedding(128, 8) for _ in range(6)])
        # Six categorical columns use learned 8-number representations. Add four
        # legal-action flags for this host and one elapsed-time feature.
        node_input_size = 6 * 8 + 4 + 1 + 4  # Four action-history features per host.
        self.encode = nn.Sequential(nn.Linear(node_input_size, hidden_units), nn.ReLU(), nn.Linear(hidden_units, hidden_units), nn.ReLU())
        self.host_head = nn.Sequential(nn.Linear(3 * hidden_units, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 4))
        self.action_ids = nn.Embedding(config.num_actions, 8)
        self.prefix_head = nn.Sequential(nn.Linear(hidden_units + 8, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 1))
        self.value = nn.Linear(hidden_units, 1)
        config = SimpleNamespace(**{**vars(config)}) if hasattr(config, '__dict__') else SimpleNamespace(observation_dim=config.observation_dim, num_actions=config.num_actions)
        config.observation_dim += config.num_actions
        fallback_width = min(64, hidden_units)
        while fallback_width > 1 and q_network_parameters(config.observation_dim, config.num_actions, fallback_width) >= 700000:
            fallback_width -= 1
        self.fallback = QNetwork(config, fallback_width)

    def configure(self, features):
        if self.layout[0].item() == 0:
            choices = infer_observation_layouts(features[:self.observation_size], self.actions)
            if len(choices) == 1:
                self.layout.copy_(torch.tensor(choices[0], dtype=torch.long, device=self.layout.device))
            elif not choices:
                self.layout[0] = -1

    def forward(self, features, action_mask):
        if self.layout[0].item() < 1:
            return self.fallback(features, action_mask)
        single_observation = features.ndim == 1
        if single_observation:
            features, action_mask = (features[None], action_mask[None])
        memory_values = features[:, self.observation_size:]
        features = features[:, :self.observation_size]
        host_count, non_host_actions, *category_counts = self.layout.tolist()
        feature_offset = 0
        category_embeddings = []
        for embedding, category_count in zip(self.categories, category_counts):
            one_hot_values = features[:, feature_offset:feature_offset + host_count * category_count].reshape(-1, host_count, category_count)
            category_ids = one_hot_values.argmax(-1) % 128
            category_embeddings.append(embedding(category_ids))
            feature_offset += host_count * category_count
        host_masks = action_mask[:, non_host_actions:].reshape(-1, 4, host_count).transpose(1, 2).float()
        time = features[:, -1:].unsqueeze(1).expand(-1, host_count, -1)
        inputs = [*category_embeddings, host_masks, time]
        inputs.append(memory_values[:, non_host_actions:].reshape(-1, 4, host_count).transpose(1, 2))
        host_vectors = self.encode(torch.cat(inputs, -1))
        adjacency = features[:, feature_offset:-1].reshape(-1, host_count, host_count)
        neighbor_mean = torch.bmm(adjacency, host_vectors) / adjacency.sum(-1, keepdim=True).clamp_min(1)
        global_mean = host_vectors.mean(1)
        context = global_mean[:, None, :].expand(-1, host_count, -1)
        host_advantages = self.host_head(torch.cat((host_vectors, neighbor_mean, context), -1))
        host_advantages = host_advantages.transpose(1, 2).reshape(-1, 4 * host_count)
        action_embeddings = self.action_ids(torch.arange(non_host_actions, device=features.device))
        action_embeddings = action_embeddings[None].expand(features.shape[0], -1, -1)
        prefix_context = global_mean[:, None, :].expand(-1, non_host_actions, -1)
        prefix_advantages = self.prefix_head(torch.cat((prefix_context, action_embeddings), -1)).squeeze(-1)
        advantage = torch.cat((prefix_advantages, host_advantages), -1)
        mean_legal_advantage = (advantage * action_mask).sum(-1, keepdim=True) / action_mask.sum(-1, keepdim=True).clamp_min(1)
        action_values = self.value(global_mean) + advantage - mean_legal_advantage
        return action_values[0] if single_observation else action_values

CURIOSITY_REWARD_WEIGHT = 0.0

class SharedHostAgent(DoubleDQNAgent):
    """Learn from runner-provided transitions, using a graph Q network."""

    def __init__(self, config):
        super().__init__(config)
        self.model = SharedHostQNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target = copy.deepcopy(self.model).eval()

    def act(self, observation, training):
        online_network = self.model
        target_network = self.target
        if training and online_network.layout[0].item() == 0:
            online_network.configure(observation.features)
            target_network.layout.copy_(online_network.layout)
        return super().act(observation, training)

    def learn(self, transition):
        super().learn(transition)


class RememberingHostAgent(SharedHostAgent):
    """Add episode-local action history to a partially observed environment.

    Each action gets a visit counter. The graph receives four counters per
    host, so it can learn whether to persist or switch after repeated attempts.
    These are observation-history features, not extra rewards or policy rules.
    The same counters are updated during training and frozen inference, and
    reset at the start of each episode. Learned weights stay in model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.action_counts = np.zeros(config.num_actions, dtype=np.float32)

    def reset(self, seed):
        super().reset(seed)
        self.action_counts.fill(0)
        self.last_observation = None

    def _with_action_history(self, observation):
        # log1p keeps repeated visits from overwhelming categorical features.
        history = np.log1p(self.action_counts).astype(np.float32)
        return SimpleNamespace(
            features=np.concatenate((observation.features, history)),
            action_mask=observation.action_mask,
        )

    def act(self, observation, training):
        self.last_observation = self._with_action_history(observation)
        action = super().act(self.last_observation, training)
        self.action_counts[action] += 1
        return action

    def learn(self, transition):
        training_transition = SimpleNamespace(
            observation=self.last_observation,
            action=transition.action,
            reward=transition.reward,
            next_observation=self._with_action_history(transition.next_observation),
            done=transition.done,
        )
        super().learn(training_transition)


def create_agent(config):
    """Train the same history-aware graph learner separately for each role."""
    return RememberingHostAgent(config)
