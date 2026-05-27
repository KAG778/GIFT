"""
PPO agent for portfolio optimization.

Stability features borrowed from TD3:
- Twin V-critics with ``min(V1, V2)`` for conservative advantage estimation
  (prevents value overestimation).
- LayerNorm for stable activation distributions.
- Value-function clipping.
- Dropout regularization on the critics.

Design notes:
- The actor emits Dirichlet concentration parameters, guaranteeing
  non-negative weights that sum to 1.
- The twin critics each estimate ``V(s)`` independently; their minimum is
  used in the GAE computation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ActorNetwork(nn.Module):
    """Actor network that outputs Dirichlet concentration parameters.

    A softplus activation keeps ``alpha > 0``; samples from the resulting
    Dirichlet are automatically normalised to portfolio weights.
    """

    def __init__(self, state_dim, hidden_dim=256, n_assets=6):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_assets)
        self.n_assets = n_assets

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        raw = self.fc3(x)
        return F.softplus(raw) + 1e-6

    def get_distribution(self, state):
        alpha = self.forward(state)
        return Dirichlet(alpha)

    def sample_weights(self, state, deterministic=False):
        if deterministic:
            alpha = self.forward(state)
            return alpha / alpha.sum(dim=-1, keepdim=True)
        dist = self.get_distribution(state)
        weights = dist.rsample()
        log_prob = dist.log_prob(weights)
        return weights, log_prob


class CriticNetwork(nn.Module):
    """Legacy single critic, kept for backward-compatible checkpoint loading."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class TwinCriticNetwork(nn.Module):
    """Twin V-critics for conservative value estimation.

    Inspired by TD3's clipped double Q-learning, adapted for PPO's V-functions.
    Using ``min(V1, V2)`` during advantage estimation prevents value
    overestimation. The two heads share architecture but train independent
    weights, each with its own LayerNorm and Dropout.
    """

    def __init__(self, state_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        # V1
        self.v1_fc1 = nn.Linear(state_dim, hidden_dim)
        self.v1_ln1 = nn.LayerNorm(hidden_dim)
        self.v1_drop1 = nn.Dropout(dropout_rate)
        self.v1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v1_ln2 = nn.LayerNorm(hidden_dim)
        self.v1_drop2 = nn.Dropout(dropout_rate)
        self.v1_fc3 = nn.Linear(hidden_dim, 1)

        # V2 (independent weights)
        self.v2_fc1 = nn.Linear(state_dim, hidden_dim)
        self.v2_ln1 = nn.LayerNorm(hidden_dim)
        self.v2_drop1 = nn.Dropout(dropout_rate)
        self.v2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v2_ln2 = nn.LayerNorm(hidden_dim)
        self.v2_drop2 = nn.Dropout(dropout_rate)
        self.v2_fc3 = nn.Linear(hidden_dim, 1)

    def _forward_v1(self, state):
        v = F.relu(self.v1_ln1(self.v1_fc1(state)))
        v = self.v1_drop1(v)
        v = F.relu(self.v1_ln2(self.v1_fc2(v)))
        v = self.v1_drop2(v)
        return self.v1_fc3(v).squeeze(-1)

    def _forward_v2(self, state):
        v = F.relu(self.v2_ln1(self.v2_fc1(state)))
        v = self.v2_drop1(v)
        v = F.relu(self.v2_ln2(self.v2_fc2(v)))
        v = self.v2_drop2(v)
        return self.v2_fc3(v).squeeze(-1)

    def forward(self, state):
        """Return (V1(s), V2(s))."""
        return self._forward_v1(state), self._forward_v2(state)

    def V1(self, state):
        """Return only V1(s) — used for value clipping reference."""
        return self._forward_v1(state)


class PPOAgent:
    """PPO agent with TD3-inspired stability features.

    The training loop is: action sampling -> GAE computation -> multiple
    mini-batch updates per rollout.
    """

    def __init__(self, state_dim, hidden_dim=256, actor_lr=3e-4,
                 critic_lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 epochs_per_update=10, batch_size=64, n_assets=6,
                 use_twin_critic=True, value_clip_epsilon=0.2,
                 dropout_rate=0.1, max_grad_norm=0.5,
                 critic_weight_decay=1e-5, seed=None):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.use_twin_critic = use_twin_critic
        self.value_clip_epsilon = value_clip_epsilon
        self.max_grad_norm = max_grad_norm

        if seed is not None:
            set_seed(seed)

        self.actor = ActorNetwork(state_dim, hidden_dim, n_assets)

        if use_twin_critic:
            self.critic = TwinCriticNetwork(state_dim, hidden_dim, dropout_rate)
        else:
            self.critic = CriticNetwork(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr,
            weight_decay=critic_weight_decay,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

    def select_action(self, state, deterministic=False):
        """Sample weights from the Dirichlet actor. If ``deterministic`` is true, return the mean."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                weights = self.actor.sample_weights(state_t, deterministic=True)
                return weights.cpu().numpy().flatten(), 0.0
            weights, log_prob = self.actor.sample_weights(state_t)
            return weights.cpu().numpy().flatten(), log_prob.item()

    def _get_values(self, states_t, next_state_t):
        """Get value estimates, handling twin- vs single-critic configurations.

        For the twin-critic case, ``min(V1, V2)`` is used as the conservative
        bootstrap value.
        """
        was_training = self.critic.training
        self.critic.eval()
        with torch.no_grad():
            if self.use_twin_critic:
                next_v1, next_v2 = self.critic(next_state_t)
                next_value = min(next_v1.item(), next_v2.item())
                v1, v2 = self.critic(states_t)
                result = (v1.cpu().numpy(), v2.cpu().numpy(), next_value, v1.detach())
            else:
                next_value = self.critic(next_state_t).item()
                values = self.critic(states_t)
                result = (values.cpu().numpy(), None, next_value, values.detach())
        if was_training:
            self.critic.train()
        return result

    def compute_gae(self, rewards, values_v1, values_v2, dones, next_value):
        """Compute Generalised Advantage Estimation.

        With twin critics, uses ``min(V1, V2)`` as the value baseline.
        """
        if values_v2 is not None:
            values = np.minimum(values_v1, values_v2)
        else:
            values = values_v1

        advantages = []
        gae = 0
        values_list = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_list[t + 1] * (1 - dones[t]) - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values_list[:-1]).to(self.device)
        return advantages, returns

    def update(self, states, actions, log_probs_old, rewards, dones, next_state):
        """PPO update: compute GAE then run multiple mini-batch passes.

        Includes:
        - Actor loss: PPO clipped objective with entropy bonus.
        - Critic loss: MSE plus optional value clipping.
        - Gradient clipping to guard against gradient explosion.
        """
        self.critic.train()

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        values_v1_np, values_v2_np, next_value, old_values = self._get_values(
            states_t, next_state_t)

        advantages, returns = self.compute_gae(
            rewards_t.cpu().numpy(), values_v1_np, values_v2_np,
            dones_t.cpu().numpy(), next_value,
        )

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = log_probs_old_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Actor loss
                dist = self.actor.get_distribution(b_states)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss
                if self.use_twin_critic:
                    v1_pred, v2_pred = self.critic(b_states)
                    v1_loss = nn.MSELoss()(v1_pred, b_returns)
                    v2_loss = nn.MSELoss()(v2_pred, b_returns)

                    if self.value_clip_epsilon > 0:
                        old_v1_batch = old_values[batch_idx]
                        v1_clipped = old_v1_batch + torch.clamp(
                            v1_pred - old_v1_batch,
                            -self.value_clip_epsilon,
                            self.value_clip_epsilon,
                        )
                        v1_clip_loss = nn.MSELoss()(v1_clipped, b_returns)
                        critic_loss = max(v1_loss, v1_clip_loss) + v2_loss
                    else:
                        critic_loss = v1_loss + v2_loss
                else:
                    values_pred = self.critic(b_states)
                    critic_loss = nn.MSELoss()(values_pred, b_returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        return {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
        }

    def save(self, path):
        """Save actor and critic weights to ``path``."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'use_twin_critic': self.use_twin_critic,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load weights from ``path``; backward-compatible with legacy single-critic checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])

        if self.use_twin_critic and 'use_twin_critic' not in checkpoint:
            # Legacy single-critic checkpoint: map old Sequential keys to V1
            print("  Warning: Loading legacy single-critic checkpoint into twin-critic agent.")
            old_state = checkpoint['critic']
            mapping = {
                'net.0.': 'v1_fc1.',
                'net.2.': 'v1_fc2.',
                'net.4.': 'v1_fc3.',
            }
            new_v1_state = {}
            for key, val in old_state.items():
                for old_prefix, new_prefix in mapping.items():
                    if key.startswith(old_prefix):
                        new_v1_state[key.replace(old_prefix, new_prefix)] = val
            self.critic.load_state_dict(new_v1_state, strict=False)
        else:
            self.critic.load_state_dict(checkpoint['critic'])
