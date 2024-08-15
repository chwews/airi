import torch
import random

from utils.task import Task, TaskPool
from utils.gpt import compute_loss
from utils.env import DarkRoom


def create_data_pool(num_tasks=64, num_episodes=1000, max_epsiode_len=20, gamma=0.99):
    taskpool = TaskPool([Task() for i in range(num_tasks)])
    taskpool.train(num_episodes=num_episodes, max_episode_len=max_epsiode_len, gamma=gamma)

    return taskpool


def train_model(model, taskpool, steps=1000, history_sample_len=50):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    losses = []

    for step in range(steps):
        optimizer.zero_grad()
        obs, actions, rewards = taskpool.sample_history(history_sample_len)
        loss = compute_loss(
            model(obs.squeeze(-1).to(device), actions.squeeze(-1).to(device), rewards.squeeze(-1).to(device))[:, :-1, :], actions.to(device)
        )
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    model.eval()

    return losses


def evaluate_ad(model, taskpool, max_steps, log=True, goal_from_tasks=True):
    if goal_from_tasks:
        env = random.choice(taskpool.tasks).env
        env.terminate_on_goal = True
    else:
        random_goal = random.choice(list(set(range(81)) - set([task.env.pos_to_state(task.env.goal_pos) for task in taskpool.tasks])))
        env = DarkRoom(size=9, goal=taskpool.tasks[0].env.state_to_pos(random_goal), random_start=False, terminate_on_goal=True)

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    initial_observation, _ = env.reset()

    observations = torch.zeros(
        (max_steps, 1),
        dtype=torch.long,
        device=device,
    )
    observations = observations.roll(-1, dims=0)
    observations[-1] = initial_observation
    observations = observations.roll(-1, dims=0)
    actions = torch.zeros(
        (max_steps, 1),
        dtype=torch.long,
        device=device,
    )
    rewards = torch.zeros(
        (max_steps, 1),
        dtype=torch.long,
        device=device
    )

    done = False
    num_steps = 0
    while not done and num_steps < max_steps:
        sliced_obs, sliced_act, sliced_rew = make_input_for_eval(observations, actions, rewards, istep=num_steps)    
        pred = model(
            obs=sliced_obs.to(device),
            act=sliced_act.to(device),
            rew=sliced_rew.to(device),
        )[:, -1]
        
        dist = torch.distributions.Categorical(logits=pred)
        action_sample = dist.sample().item()
        
        next_observation, reward, done, _, _ = env.step(action_sample)
        
        if log:
            print(f"Predicted Action: {action_sample}, Reward: {reward}, Next Observation: {env.state_to_pos(next_observation)}, Done: {done}, Goal: {env.goal_pos}")
        
        observations = actions.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)

        observations[-1] = next_observation
        actions[-1] = action_sample
        rewards[-1] = reward
        num_steps += 1

    return done


def make_input_for_eval(
    observations,
    actions,
    rewards,
    istep,
):
    if istep == 0:
        num_envs = actions.shape[1]
        inp = (
            observations.T[:, -(istep + 1):],
            torch.empty(
                num_envs,
                0,
                dtype=actions.dtype,
                device=actions.device,
            ),
            torch.empty(
                num_envs,
                0,
                dtype=rewards.dtype,
                device=rewards.device
            ),
        )
    else:
        inp = (observations.T[:, -(istep + 1):], actions.T[:, -istep:], rewards.T[:, -istep:])

    return inp
