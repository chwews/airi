import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib.animation import FuncAnimation, PillowWriter
from torch.distributions import Categorical
from tqdm import trange
from acrl import Actor, Critic
from env import DarkRoom

class Task:

    def __init__(self, env=None,):
        if env is None:
            self.env = DarkRoom(size=9, goal=None, random_start=False, terminate_on_goal=False,)
        else:
            self.env = env

        self.actor = Actor(self.env.observation_space.n, self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.n)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        
    def train(self, num_episodes=1000, max_episode_len=20, gamma=0.99):             
        buffer_size = num_episodes * max_episode_len
        n_envs = 1
        
        self.pos = 0
        self.observations = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.actor.train()
        self.critic.train()
        
        for episode in trange(num_episodes):
            returns = self._run_episode(max_episode_len, gamma)
    
            returns = torch.tensor(returns, dtype=torch.float32)
        
            obs = self.observations[self.pos - max_episode_len:self.pos]
            length = obs.shape[0]
            obs = obs.reshape((length, 1))
            observations_tensor = torch.tensor((obs == np.tile(np.arange(self.env.observation_space.n), (length, 1))), dtype=torch.float32)
            
            actions_tensor = torch.tensor(self.actions[self.pos - max_episode_len:self.pos], dtype=torch.int64)

            values = self.critic(observations_tensor).squeeze()
            critic_loss = nn.MSELoss()(values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            action_probs = self.actor(observations_tensor)
            action_log_probs = torch.log(action_probs.gather(1, actions_tensor).squeeze())
            advantages = returns - values.detach()
            actor_loss = - (advantages * action_log_probs).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        self.actor.eval()
        self.critic.eval()
        
    def save_animation(self, filename='animation.gif', interval=200):
        fig, ax = plt.subplots()

        frames = []
        for state in self._infer():
            self.env.agent_pos = self.env.state_to_pos(state)
            frame = self.env.render()
            frames.append(frame)

        img = ax.imshow(frames[0])

        def update(frame):
            img.set_data(frame)
            return img,

        ani = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False, interval=interval)
        writer = PillowWriter(fps=1000//interval)
        ani.save(filename, writer=writer)
        
    def sample_history(self, length=40):
        if length > self.observations.shape[0]:
            raise IndexError("History contains fewer samples than necessary.")

        start = random.randint(0, self.observations.shape[0] - length)
        return (
            self.observations[start:start + length].reshape((-1, 1)).astype(int),
            self.actions[start:start + length].reshape((-1, 1)).astype(int),
            self.rewards[start:start + length].reshape((-1, 1)).astype(float),
        )

    def _run_episode(self, max_len=20, gamma=0.99):
        state, _ = self.env.reset()

        done = False
        current_len = 0
        returns = []
        R = 0 
        
        while not done and current_len < max_len:
            state_one_hot = self._one_hot_encode(np.array([state]))
            
            action_probs = self.actor(torch.tensor(state_one_hot.reshape(1, self.env.observation_space.n), dtype=torch.float32))
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()

            next_state, reward, done, _, _ = self.env.step(action)

            self.observations[self.pos] = np.array([state])
            self.actions[self.pos] = np.array([action])
            self.rewards[self.pos] = np.array([reward])

            state = next_state

            R = reward + gamma * R
            returns.insert(0, R)
            
            self.pos += 1
            current_len += 1
            
        return returns
    

    def _one_hot_encode(self, buffer):
        length = buffer.shape[0]
        buffer = buffer.reshape((length, 1))
        return (buffer == np.tile(np.arange(self.env.observation_space.n), (length, 1))).astype(float)

    def _infer(self, max_len=20):
        self.env.terminate_on_goal = True
        states = []
        
        state, _ = self.env.reset()

        done = False
        current_len = 0
        
        self.actor.eval()
        
        while not done and current_len < max_len:
            state_one_hot = self._one_hot_encode(np.array([state]))
            
            action_probs = self.actor(torch.tensor(state_one_hot.reshape(1, self.env.observation_space.n), dtype=torch.float32))
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()

            next_state, _, done, _, _ = self.env.step(action)
            
            states.append(state)
            state = next_state
            
            current_len += 1
            
        states.append(state)
        
        self.env.terminate_on_goal = False
        self.actor.train()

        return states
    
class TaskPool:

    def __init__(self, tasks):
        if not tasks:
            raise ValueError("The task list cannot be empty.")
        for task in tasks[1:]:
            if task.env.observation_space.n != tasks[0].env.observation_space.n:
                raise ValueError("All tasks must have the same environment observation dimension.")

        self.tasks = tasks
        
    def train(self, num_episodes=1000, max_episode_len=20, gamma=0.99):
        for task in self.tasks:
            task.train(num_episodes, max_episode_len, gamma)
            
    def sample_history(self, length=40):
        samples = [task.sample_history(length) for task in self.tasks]

        observations, actions, rewards = zip(*samples)

        return (
            torch.tensor(observations, dtype=torch.long, device='cpu'),
            torch.tensor(actions, dtype=torch.long, device='cpu'),
            torch.tensor(rewards, dtype=torch.long, device='cpu') 
        )

    def goals(self):
        return [task.env.pos_to_state(task.env.goal_pos) for task in self.tasks]