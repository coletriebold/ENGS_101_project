"""Five-step prioritized dueling Double DQN, written for readability.

To use this standalone file, copy its contents into student_agent.py and run:
    python run.py --steps 10000 --eval-episodes 3 --device cpu

Both roles use the same settings and architecture, with separately trained
weights. Model dimensions and device come from the course configuration.
"""
from collections import deque
import copy
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# These settings are shared by Red and Blue. The runner trains their weights
# separately. Changing a setting here changes it for both roles.
HIDDEN_UNITS = 128
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
RETURN_STEPS = 5
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
    """A fully connected layer learns one weight per connection and one bias
    per output neuron. For example, Linear(3, 2) has 3*2 weights + 2 biases.
    """
    weights = input_units * output_units
    biases = output_units
    return weights + biases


def q_network_parameters(observation_size, action_count, hidden_units):
    """Count each layer separately so we can respect the course model limit."""
    # The network sees both the observation and which actions are legal.
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
    """Usually use 128 neurons; shrink only if a larger environment needs it."""
    for hidden_units in range(HIDDEN_UNITS, 0, -1):
        parameter_count = q_network_parameters(
            config.observation_dim, config.num_actions, hidden_units
        )
        if parameter_count <= parameter_budget:
            return hidden_units
    raise ValueError('Observation/action dimensions exceed the model budget')


class QNetwork(nn.Module):
    """Estimate how valuable each legal action is.

    'Dueling' means learning a state value and a separate advantage for each
    action. Combining them produces the Q value used to choose an action.
    """

    def __init__(self, config, hidden_units):
        super().__init__()
        input_units = config.observation_dim + config.num_actions
        self.body = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        # Keep these layer names so existing checkpoints still load.
        self.value = nn.Linear(hidden_units, 1)
        self.advantage = nn.Linear(hidden_units, config.num_actions)

    def forward(self, features, action_mask):
        network_input = torch.cat((features, action_mask.float()), dim=-1)
        hidden = self.body(network_input)
        state_value = self.value(hidden)
        advantages = self.advantage(hidden)

        # Center advantages using legal actions only. clamp_min avoids division
        # by zero; the course interface normally guarantees a legal action.
        legal_advantage_sum = (advantages * action_mask).sum(dim=-1, keepdim=True)
        legal_action_count = action_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        average_advantage = legal_advantage_sum / legal_action_count
        return state_value + advantages - average_advantage


class Experience(NamedTuple):
    """One replay entry, with names instead of an unexplained tuple of values.

    accumulated_reward covers up to RETURN_STEPS steps. future_discount is
    gamma raised to that number of steps, or zero if the episode ended.
    The last two fields are used only by the Monte Carlo extension.
    """
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
    """Remember experiences and revisit surprising ones more often."""

    def __init__(self, config):
        # Approximate storage for two float32 observations, two Boolean masks,
        # and scalar fields. Bound array storage for larger hidden scenarios.
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


def create_agent(config):
    """Course entry point: the same learner supports either role."""
    return DoubleDQNAgent(config)
