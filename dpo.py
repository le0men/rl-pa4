import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import validate_model
from model import ContinuousPolicy
import gymnasium as gym
from tqdm import tqdm
from batch_collection import collect_pair_data
import argparse
import yaml
import matplotlib.pyplot as plt
class PairedTrajectoryDataset(Dataset):
    def __init__(self, pair_data):
        self.data = pair_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        t1, t2 = p["traj1"], p["traj2"]
        return {
            "traj1_state": torch.from_numpy(t1[0]).float(),  # (T, obs_dim)
            "traj1_act": torch.from_numpy(t1[2]).float(),  # (T, act_dim)
            "traj1_logp": torch.from_numpy(t1[3]).float(),
            "traj2_state": torch.from_numpy(t2[0]).float(),
            "traj2_act": torch.from_numpy(t2[2]).float(),
            "traj2_logp": torch.from_numpy(t2[3]).float(),
            "label": torch.tensor(p["label"], dtype=torch.float32),
        }

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}


class DPOTrainer:
    def __init__(self, env, policy, optimizer, beta=1, batch_size=16, device="cpu"):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.beta = beta
        self.batch_size = batch_size
        self.device = device

    def _evaluate(self, seed, n_trajs=40):
        mean_rew, std_rew = validate_model(self.policy, self.env, n_trajs)
        return mean_rew, std_rew

    def train(self, pair_data, num_epochs_per_iter=6, num_iterations=10, seed=None):
        # TODO: implement DPO training
        dataset = PairedTrajectoryDataset(pair_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle= True)
        
        losses_agg = []

        for iteration in range(num_iterations):
            # if not first iteration, do iterative DPO
            # otherwise, just do normal DPO
            if iteration==0:
                for _ in tqdm(range(num_epochs_per_iter)):    
                    losses_graph = []          
                    for batch in loader:
        
                        #go through traj (stored s,a) and calc theta logprobs
                        theta_w_logp = []
                        theta_l_logp = []
                        ref_w_logp = []
                        ref_l_logp = []

                        for i in range(len(batch["traj1_state"])):
                            states1 = batch["traj1_state"][i]
                            acts1 = batch["traj1_act"][i]
                            states2 = batch["traj2_state"][i]
                            acts2 = batch["traj2_act"][i]
                            label = batch["label"][i]
                            index = torch.multinomial(label, num_samples=1).item()
                            
                            theta_traj1_logp = self.policy.compute_log_likelihood(states1, acts1)
                            theta_traj2_logp = self.policy.compute_log_likelihood(states2, acts2)

                            if index==0:
                                theta_w_logp.append(torch.sum(theta_traj1_logp))
                                theta_l_logp.append(torch.sum(theta_traj2_logp))
                                ref_w_logp.append(batch["traj1_logp"][i])
                                ref_l_logp.append(batch["traj2_logp"][i])
                            else:
                                theta_l_logp.append(torch.sum(theta_traj1_logp))
                                theta_w_logp.append(torch.sum(theta_traj2_logp))
                                ref_l_logp.append(batch["traj1_logp"][i])
                                ref_w_logp.append(batch["traj2_logp"][i])
                          

                        #use with stored data logprobs to calcv loss
                        theta_w_logp = torch.stack(theta_w_logp)
                        theta_l_logp = torch.stack(theta_l_logp)
                        ref_w_logp = torch.stack(ref_w_logp)
                        ref_l_logp = torch.stack(ref_l_logp)

                        logits_loss = self.beta * (theta_w_logp - theta_l_logp - (ref_w_logp - ref_l_logp))
                        losses = (logits_loss-1)**2
                        
                        # losses = -torch.nn.functional.logsigmoid(logits_loss)
                        loss = losses.mean()
                        losses_graph.append(loss.item())
                        

                        #gd on lloss
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
 
                    losses_agg.append(sum(losses_graph)/self.batch_size)
                    print(losses_agg)    

            else:
                pass

        plt.plot(losses_agg)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


def main():
    env = gym.make("Swimmer-v5")
    policy = ContinuousPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    # load model
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))
    pair_data = torch.load("pair_data.pt", weights_only=False)

    # argparse
    # load hparams
    with open("hparam.yaml", "r") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(hparams["lr"]))
    dpo = DPOTrainer(env, policy, optimizer, float(hparams["beta"]), int(hparams["batch_size"]))
    
    if hparams["iterative_dpo"]:
        iterations = 10
    else:
        iterations = 1

    dpo.train(pair_data, num_iterations=iterations, seed=42, num_epochs_per_iter=hparams["num_epochs_per_iter"])

    if hparams["iterative_dpo"]:
        pass
    else:
        torch.save(policy.state_dict(), "dpo.pt")

    
    print(dpo._evaluate(seed=42))

if __name__ == "__main__":
    main()
