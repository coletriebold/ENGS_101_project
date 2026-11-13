"""Double DQN with RND exploration and Monte Carlo return assistance.

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


# Monte Carlo + RND additions. The Double DQN foundation above is identical
# to double_dqn_agent.py; only the training reward and value target differ.
RND_REWARD_WEIGHT = 0.005
RND_LEARNING_RATE = 0.0003
Q_NETWORK_BUDGET = 760_000
RND_PARAMETER_BUDGET = 220_000


def choose_novelty_hidden_units(observation_size):
    """Reserve room for both the fixed random network and its predictor."""
    for hidden_units in range(64, 0, -1):
        # Target: observation -> hidden layer -> 32-dimensional output.
        target_parameters = (
            linear_layer_parameters(observation_size, hidden_units)
            + linear_layer_parameters(hidden_units, 32)
        )
        # Predictor: observation -> hidden -> hidden -> same output size.
        predictor_parameters = (
            linear_layer_parameters(observation_size, hidden_units)
            + linear_layer_parameters(hidden_units, hidden_units)
            + linear_layer_parameters(hidden_units, 32)
        )
        # Preserve the original conservative sizing rule, including its margin.
        safety_margin = hidden_units
        estimated_parameters = target_parameters + predictor_parameters + safety_margin
        if estimated_parameters <= RND_PARAMETER_BUDGET:
            return hidden_units
    raise ValueError('Observation dimensions exceed the novelty network budget')


class Novelty(nn.Module):
    """Random Network Distillation: unfamiliar observations are harder to predict.

    The fixed random network never learns. A second network learns to match
    its outputs. Their prediction error supplies a small exploration reward.
    Layer/buffer names are retained for compatibility with saved checkpoints.
    """

    def __init__(self, observation_size):
        super().__init__()
        hidden_units = choose_novelty_hidden_units(observation_size)
        self.fixed = nn.Sequential(
            nn.Linear(observation_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 32),
        )
        self.fixed.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(observation_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 32),
        )
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=RND_LEARNING_RATE
        )

        # Buffers are running statistics, not trainable weights. Registering them
        # ensures they move with the model and are included in checkpoints.
        self.register_buffer('obs_count', torch.tensor(0.0))
        self.register_buffer('obs_mean', torch.zeros(observation_size))
        self.register_buffer('obs_m2', torch.zeros(observation_size))
        self.register_buffer('error_sq', torch.tensor(0.0))
        self.register_buffer('error_count', torch.tensor(0.0))
        self.register_buffer('bonus_sum', torch.tensor(0.0))

    def bonus(self, features):
        novelty_features = features.detach().clone()
        # The final feature is elapsed time. Clock advancement alone should not
        # make a state novel. The Q network still sees the original time feature.
        novelty_features[-1] = 0.0
        with torch.no_grad():
            # Update the observation mean and variance without storing history.
            self.obs_count.add_(1)
            difference_from_mean = novelty_features - self.obs_mean
            self.obs_mean.add_(difference_from_mean / self.obs_count)
            self.obs_m2.add_(
                difference_from_mean * (novelty_features - self.obs_mean)
            )
            variance = self.obs_m2 / self.obs_count.clamp_min(1.0)
            standard_deviation = variance.clamp_min(0.01).sqrt()
            normalized_features = (
                (novelty_features - self.obs_mean) / standard_deviation
            ).clamp(-5, 5)
            desired_output = self.fixed(normalized_features)
            predicted_output = self.predictor(normalized_features)
            prediction_error = (predicted_output - desired_output).square().mean()

        # Learn to recognize this observation. Use its pre-update error as the
        # bonus, so newly visited observations are rewarded before learning them.
        predictor_loss = (
            self.predictor(normalized_features) - desired_output
        ).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        predictor_loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Normalize against typical error size to keep the bonus bounded.
            self.error_count.add_(1)
            self.error_sq.add_(prediction_error.square())
            typical_error = (self.error_sq / self.error_count).sqrt().clamp_min(1e-8)
            normalized_bonus = (prediction_error / typical_error).clamp(0, 5)
            self.bonus_sum.add_(normalized_bonus)
        return normalized_bonus.item()


class AuxiliaryModel(nn.Module):
    """Keep Q-network weights and curiosity statistics in one saved model."""

    def __init__(self, core, novelty):
        super().__init__()
        self.core = core
        self.novelty = novelty

    def forward(self, features, action_mask):
        # Greedy evaluation needs only the Q network; curiosity learns in learn().
        return self.core(features, action_mask)


class MonteCarloRNDAgent(DoubleDQNAgent):
    """Use completed episodes to help propagate successful outcomes backward.

    Once an episode ends, compute the discounted return from each action all
    the way to the episode's end. For replay entries with a known return, use
    whichever target is larger: that observed return or the Double-DQN target.
    This is an optimistic heuristic, not a proven bound on expected value.
    """

    def __init__(self, config):
        # Leave enough of the million-parameter limit for the two RND networks.
        super().__init__(config, parameter_budget=Q_NETWORK_BUDGET)
        novelty = Novelty(config.observation_dim).to(self.device)
        self.model = AuxiliaryModel(self.model, novelty).to(self.device)
        self.target = copy.deepcopy(self.model).eval()
        self.episode_rewards = []
        self.episode_replay_entries = []

    def reset(self, seed):
        super().reset(seed)
        self.episode_rewards = []
        self.episode_replay_entries = []

    def _store_oldest_transition(self):
        first_transition = self.pending_transitions[0]
        experience = self._build_experience()
        replay_slot = self.replay.cursor
        self.replay.add(experience)
        # Remember where this step was stored so its full return can be filled
        # in later. Retain object identity to detect overwritten replay slots.
        self.episode_replay_entries.append(
            (replay_slot, experience, first_transition.episode_index)
        )
        self.pending_transitions.popleft()

    def _attach_completed_episode_returns(self):
        # Work backward: G[t] = reward[t] + gamma * G[t+1].
        episode_returns = np.zeros(len(self.episode_rewards), dtype=np.float32)
        discounted_return = 0.0
        for step in range(len(self.episode_rewards) - 1, -1, -1):
            discounted_return = (
                self.episode_rewards[step] + DISCOUNT_FACTOR * discounted_return
            )
            episode_returns[step] = discounted_return

        for replay_slot, experience, episode_step in self.episode_replay_entries:
            if self.replay.data[replay_slot] is experience:
                self.replay.data[replay_slot] = experience._replace(
                    episode_return=float(episode_returns[episode_step]),
                    return_is_known=True,
                )

    def _calculate_targets(self, batch):
        double_dqn_targets = super()._calculate_targets(batch)
        with torch.no_grad():
            observed_returns = batch.episode_return.float()
            assisted_targets = torch.maximum(double_dqn_targets, observed_returns)
            return torch.where(
                batch.return_is_known, assisted_targets, double_dqn_targets
            )

    def learn(self, transition):
        self.training_steps += 1
        next_features = torch.as_tensor(
            transition.next_observation.features, device=self.device
        )
        novelty_bonus = self.model.novelty.bonus(next_features)
        learning_reward = float(transition.reward) + RND_REWARD_WEIGHT * novelty_bonus

        # Build a separate training transition; leave the environment's original
        # reward and its score untouched. Full returns include this same bonus.
        training_transition = SimpleNamespace(
            observation=transition.observation,
            action=transition.action,
            reward=learning_reward,
            next_observation=transition.next_observation,
            done=transition.done,
            episode_index=len(self.episode_rewards),
        )
        self.episode_rewards.append(learning_reward)
        self._remember_transition(training_transition)
        if transition.done:
            self._attach_completed_episode_returns()
        self._train_from_replay()


def create_agent(config):
    """Course entry point: the same learner supports either role."""
    return MonteCarloRNDAgent(config)
