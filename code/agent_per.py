import numpy as np
import random
from collections import namedtuple, deque
from Memory import Memory

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(10000)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = Memory(BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Here we'll deal with the empty memory problem: we pre-populate our memory 
        # by taking random actions and storing the experience.
        self.tree_idx = None
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory


    
        e = self.experience(state, action, reward, next_state, done)
        self.memory.store(e)

        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
               
            # Obtain random mini-batch from memory
            self.tree_idx, batch, ISWeights_mb = self.memory.sample(BATCH_SIZE)
            
            states = torch.from_numpy(np.vstack([each[0][0] for each in batch])).float().to(device)
            actions = torch.from_numpy(np.vstack([each[0][1] for each in batch])).long().to(device)
            rewards = torch.from_numpy(np.stack([[each[0][2]] for each in batch])).float().to(device)
            next_states = torch.from_numpy(np.vstack([each[0][3] for each in batch])).float().to(device)
            dones = torch.from_numpy(np.stack([[each[0][4]] for each in batch]).astype(np.uint8)).float().to(device)

            experiences = (states, actions, rewards, next_states, dones)
            
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):

        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    
    def learn(self, experiences, gamma):
        

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
       
        # Choose actions according to local network
        
        next_actions = self.qnetwork_local(next_states).argmax(dim=1)
        
        # Choose values from target network 
        Q_targets_next = self.qnetwork_target(next_states).detach()[np.arange(BATCH_SIZE),next_actions].unsqueeze(1)


        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # Update memory having the batch loss as priority value
        batch_loss = np.ones(BATCH_SIZE)*loss.data.cpu().numpy()
      
        self.memory.batch_update(self.tree_idx, batch_loss)
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)         


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

