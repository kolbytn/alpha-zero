import sys
import socket
import time
import random
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ReversiEnv import ReversiEnv


class MCNode:
    def __init__(self, state, player, parent, action):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        
        self.value = 0
        self.probs = None
        self.visits = 0


class Network(nn.Module):
    def __init__(self, state_size, action_size, channels=512, drop=.3):
        super().__init__()
        self.action_size = action_size
        self.board_size = state_size[1]
        self.channels = channels
        in_c = state_size[0]
        
        self.conv = nn.Sequential(nn.Conv2d(in_c, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(channels * self.board_size * self.board_size, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Dropout(drop),
                                    nn.Linear(1024, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Dropout(drop)) 

        self.probs_layer = nn.Sequential(nn.Linear(512, action_size),
                                         nn.Softmax(1))

        self.value_layer = nn.Sequential(nn.Linear(512, 1),
                                         nn.Tanh())
        
    def forward(self, x):
        conv_out = self.conv(x)
        lin_in = conv_out.view(-1, self.channels * self.board_size * self.board_size)
        lin_out = self.linear(lin_in)
        p = self.probs_layer(lin_out)
        v = self.value_layer(lin_out)
        return p.squeeze(), v.squeeze()


class RLDataset(Dataset):
    def __init__(self, data):
        super(RLDataset, self).__init__()
        self.data = data
  
    def __getitem__(self, index):
        return self.data[index]
 
    def __len__(self):
        return len(self.data)


class AlphaAgent:
    def __init__(self, num_sims=20, train_epochs=10, batch_size=64, lr=1e-3, 
                 cpuct=math.sqrt(2), model_path='models/', device='cuda'):
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.device = device
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.model_path = model_path

        self.net = Network((2, 8, 8), 64).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_action(self, state, player):
        root = MCNode(np.array(state), player, None, None)

        for _ in range(self.num_sims):
            self.search(root)

        probs = np.zeros((64), dtype=np.float32)
        for c in root.children:
            probs[self.get_action_index(c.action)] = c.visits
        probs /= np.sum(probs)

        action = np.random.choice(64, p=probs)
        action = [action % 8, action // 8]

        return action, probs

    def search(self, node):
        if ReversiEnv.is_game_over(node.state):
            v = ReversiEnv.get_winner(node.state, node.player)
            return -v

        if node.probs is None:
            valids = ReversiEnv.get_valid_actions(node.state, node.player)
            valids = [self.get_action_index(x) for x in valids]
            valids_mask = np.zeros((64))
            valids_mask[valids] = 1

            with torch.no_grad():
                self.net.eval()
                t_state = self.prepare_state(node.state, node.player)
                probs, v = self.net(t_state)

                node.probs = probs.cpu().numpy()
                node.probs *= valids_mask
                if np.sum(valids_mask) > 0:
                    if np.sum(node.probs) == 0:
                        print('No prob for valid moves')
                        node.probs = valids_mask / np.sum(valids_mask)
                    else:
                        node.probs /= np.sum(node.probs)

            return -v.item()

        if len(node.children) == 0:
            actions = ReversiEnv.get_valid_actions(node.state, node.player)

            for a in actions:
                next_state = ReversiEnv.get_next_state(node.state, a, node.player)
                child = MCNode(next_state, ReversiEnv.get_next_player(node.player), node, a)
                node.children.append(child)
            
            if len(node.children) == 0:
                node.children.append(MCNode(node.state, ReversiEnv.get_next_player(node.player), node, None))

        scores = [c.value + self.cpuct * c.parent.probs[self.get_action_index(c.action)] * np.sqrt(node.visits) / (1 + c.visits) 
                  for c in node.children]

        child = node.children[scores.index(max(scores))]
        v = self.search(child)

        if child.value == 0:
            child.value = v
        else:
            child.value = (child.visits * child.value + v) / (child.visits + 1)
        child.visits += 1

        return -v

    def get_action_index(self, action):
        if action is None:
            return 0
        return action[0] + action[1] * 8

    def prepare_state(self, state, player):

        if state.shape == (8, 8):
            t_state = torch.zeros((1, 2, 8, 8), dtype=torch.float, device=self.device)
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state[i][j] == player:
                        t_state[0][0][i][j] = 1
                    elif state[i][j] == ReversiEnv.get_next_player(player):
                        t_state[0][1][i][j] = 1
        elif len(state.shape) == 3 and state.shape[1] == 8 and state.shape[2] == 8:
            t_state = torch.zeros((state.shape[0], 2, 8, 8), dtype=torch.float, device=self.device)
            for b in range(state.shape[0]):
                for i in range(state.shape[1]):
                    for j in range(state.shape[2]):
                        if state[b][i][j] == player[b]:
                            t_state[b][0][i][j] = 1
                        elif state[b][i][j] == ReversiEnv.get_next_player(player[b]):
                            t_state[b][1][i][j] = 1
        else:
            raise Exception("Invalid state size", state)

        return t_state

    def save_model(self, n):
        torch.save(self.net.state_dict(), self.model_path + 'net' + str(n))

    def load_model(self, n):
        self.net.load_state_dict(torch.load(self.model_path + 'net' + str(n)))

    def learn(self, memory):
        self.net.train()
        dataset = RLDataset(memory)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for _ in range(self.train_epochs):
            for state, player, pi, v in loader:
                self.optim.zero_grad()

                pi, v = pi.to(self.device), v.to(self.device)
                player = player.float()

                t_state = self.prepare_state(state, player)
                pi_pred, v_pred = self.net(t_state)

                loss_pi = -torch.mean(pi * pi_pred)
                loss_v = torch.mean((v - v_pred) ** 2)

                loss = loss_pi + loss_v

                loss.backward()
                self.optim.step()
