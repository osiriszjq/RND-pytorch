import torch
import numpy as np
import random
from torch.distributions.categorical import Categorical

from segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, R):
        data = (obs_t, action, R)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obses_t, actions, returns = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

class SILModel:
    def __init__(self, model, args, optimizer, device,
        sil_capacity=5000, batch_size = 512, min_batch_size =32,
        sil_alpha=0.6, sil_beta=0.1, sil_n_update=4, w_value=0.01):
        self.args = args
        self.sil_capacity = sil_capacity
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.sil_alpha = sil_alpha
        self.sil_beta = sil_beta
        self.sil_n_update = sil_n_update
        self.w_value = w_value
        self.model = model
        #self.rnd = rnd
        self.optimizer = optimizer
        self.device = device
        self.running_episodes = [[] for _ in range(self.args.num_worker)]
        self.buffer = PrioritizedReplayBuffer(self.sil_capacity, self.sil_alpha)

    def step(self, obs, actions, rewards, dones):
        for n in range(self.args.num_worker):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n]])

        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)

    def add_episode(self, trajectory):
        obs, actions, rewards = [], [], []
        for (ob, action, reward) in trajectory:
            obs.append(np.float32(ob) / 255.)
            actions.append(action)
            rewards.append(max(min(reward,1),-1))
        returns = self.discount(rewards, self.args.ext_gamma)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer.add(ob, action, R)

    def discount(self, rewards, gamma):
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = gamma*running_reward + rewards[t]
            rewards[t] = running_reward
        return rewards

    def sample_batch(self, batch_size):
        if len(self.buffer) >= 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.sil_beta)
        else:
            return None, None, None, None, None

    def train(self):
        #print('buffer size:', len(self.buffer))
        if len(self.buffer) < 100:
            return 0
        total_valid_samples = 0
        for n in range(self.sil_n_update):
            states, actions, rewards, weights, idxes = self.sample_batch(self.batch_size)
            num_valid_samples = 0
            if states is not None:
                states = torch.FloatTensor(states).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
                actions = torch.LongTensor(actions).to(self.device)
                weights = torch.FloatTensor(weights).to(self.device).view(-1, 1)

                actions_probs, value_ext, value_int = self.model(states)
                m = Categorical(actions_probs)
                entropy = m.entropy().mean()
                log_prob = -m.log_prob(actions).view(-1, 1)
                clipped_log_prob = log_prob.clamp(max=5)

                advantages = (rewards - value_ext).detach()
                clipped_advantages = advantages.clamp(min=0, max=1)
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                num_valid_samples = masks.sum().item()
                if num_valid_samples == 0:
                    break
                num_samples = max(num_valid_samples, self.min_batch_size)
                total_valid_samples += num_valid_samples

                masks = torch.FloatTensor(masks).to(self.device)
                #mean_adv = (torch.sum(clipped_advantages) / num_valid_samples).item()
                action_loss = torch.sum(weights * clipped_log_prob * clipped_advantages) / num_samples
                entropy_reg = torch.sum(weights * entropy * masks) / num_samples
                policy_loss = action_loss - self.args.entropy_coef * entropy_reg
                delta = torch.clamp(value_ext - rewards, -1, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value_ext * delta) / num_samples
                loss = policy_loss + 0.5 * self.w_value * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().detach().numpy())
        return total_valid_samples
