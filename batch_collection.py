import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import itertools
from utils import device
from model import ContinuousPolicy


def batch_collection(env, policy, seed, *, total_trajectories=16, smoothing=False):
    """Collect `total_trajectories` episodes from a SyncVectorEnv."""
    trajectories = []

    #TODO: implement batch_collection``
    obs,info = env.reset()
    obs = torch.from_numpy(obs).float()
    
    pbar = tqdm(total=(total_trajectories//env.num_envs))
    while len(trajectories) < total_trajectories:
        dones = [False]
        curr_trajs = []
        for i in range(env.num_envs):
            curr_trajs.append([[],[],[],0])

        #while not any(dones):
        for _ in range(1000):
            action = policy.sample_action(obs,smooth=smoothing).detach()
            observations, rewards, term, trunc, infos = env.step(action.detach().numpy())
            dones = torch.from_numpy(np.logical_or(term, trunc))
            logps = policy.compute_log_likelihood(obs,action).detach().numpy()
            
            for i in range(env.num_envs):
                traj = curr_trajs[i]
                traj[0].append(obs[i])
                traj[1].append(rewards[i])
                traj[2].append(action[i])
                traj[3]= traj[3]+logps[i]

            obs=observations
            obs = torch.from_numpy(obs).float()

        for i in range(env.num_envs):
                traj = curr_trajs[i]
                traj[0] = np.array(traj[0])
                traj[1] = np.array(traj[1])
                traj[2] = np.array(traj[2])
                traj[3] = np.array(traj[3])

        pbar.update(1)

        trajectories.extend(curr_trajs)

    pbar.close()
    return trajectories[:total_trajectories]

def pair_trajectories(trajs, temp=1.0, seed=None):
    rng = np.random.default_rng(seed)
    returns = np.array([np.sum(t[1]) for t in trajs])

    idx_pairs = list(zip(rng.permutation(len(trajs)), rng.permutation(len(trajs))))
    idx_pairs = [pair for pair in idx_pairs if pair[0] != pair[1]]

    pair_data = []
    #TODO: implement pair_trajectories
    # probaility of ppickin either one
    for i1,i2 in idx_pairs:
        label1 = 1/(1+ np.exp(-(returns[i1]-returns[i2])))
        label2 = 1/(1+ np.exp(-(returns[i2]-returns[i1])))

        pair_data.append({"traj1": trajs[i1], "traj2" : trajs[i2], "label" : (label1,label2)})

    return pair_data

def collect_pair_data(policy, seed, total_trajectories=16, smoothing=False):
    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(16)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    trajectories = batch_collection(env, policy, seed, total_trajectories=total_trajectories, smoothing=smoothing)
    return pair_trajectories(trajectories, seed=seed)

def main():
    num_envs = 32 # set this based on your hardware
    seed = 42

    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(num_envs)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    policy = ContinuousPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))

    trajectories = batch_collection(env, policy, seed, total_trajectories=5000)

    mean_reward = np.mean([np.sum(traj[1]) for traj in trajectories])
    std_reward = np.std([np.sum(traj[1]) for traj in trajectories])
    print(f"Mean reward of trajectories: {mean_reward}, std: {std_reward}")
    pair_data = pair_trajectories(trajectories, seed=seed)
    torch.save(pair_data, "pair_data.pt")

if __name__ == "__main__":
    main()
