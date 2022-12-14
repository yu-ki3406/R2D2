import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from time import sleep
import os
from model import Q_Base_Net
from replay_memory import NStepMemory, ReplayMemory
from torch.utils.tensorboard import SummaryWriter

def learner_process(n_actor, shared_dict):
    learner = Learner(n_actor, shared_dict)
    learner.run()


class Learner:
    def __init__(self, n_actor, shared_dict, device="cuda"):
        self.gamma = 0.99
        self.alpha = 0.6
        self.bootstrap_step = 3
        self.initial_exploration = 50000
        self.priority_epsilon = 1e-6
        self.device = device
        self.n_epochs = 0
        self.n_actor = n_actor
        self.memory_path = os.path.join(
            './', 'logs', 'memory'
        )
        self.summary_path = os.path.join(
            './', 'logs', 'summary'
        )
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.writer = SummaryWriter(log_dir=self.summary_path)
        self.loss_step = 0


        self.burn_in_length = 10
        self.learning_length = 10
        self.sequence_length = self.burn_in_length + self.learning_length

        self.memory_size = 500000
        self.batch_size = 8
        self.memory_load_interval = 20
        self.reply_memory = ReplayMemory(self.memory_size, self.batch_size, self.bootstrap_step)

        self.shared_dict = shared_dict
        self.net_save_interval = 100
        self.target_update_interval = 1000

        self.net = Q_Base_Net(self.device, action_num=19).to(self.device)
        self.target_net = Q_Base_Net(self.device, action_num=19).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.save_model()
        self.optim = optim.RMSprop(self.net.parameters(), lr=0.00025 / 4, alpha=0.95, eps=1.5e-7,
                                   centered=True)

    def run(self):
        while True:
            if self.reply_memory.size > self.initial_exploration:
                self.train()
                self.n_epochs += 1
                if self.n_epochs % 100 == 0:
                    print('trained{}epochs').format(self.n_epochs)

                self.interval()

            else:

                for i in range(self.n_actor):
                    try:
                        self.reply_memory.load(self.memory_path, i)
                    except:
                        pass

    def train(self):
        batch, seq_index, index = self.replay_memory.sample(self.device)
        self.net.set_state(batch['hs'], batch['cs'])
        self.target_net.set_state(batch['target_hs'], batch['target_cs'])

        state = batch['state'][:self.burn_in_length]
        next_state = batch['next_state'][:self.burn_in_length]
        with torch.no_grad():
            _ = self.net(state)
            _ = self.target_net(next_state)

        state = batch['state'][self.burn_in_length:]
        next_state = batch['next_state'][self.burn_in_length:]

        q_value = self.net(state).gather(1, batch['action'].view(-1, 1))

        with torch.no_grad():
            next_action = torch.argmax(
                self.net(next_state), 1).view(-1, 1)
            next_q_value = self.target_net(
                next_state).gather(1, next_action)
            target_q_value = batch["reward"].view(-1, 1) + (
                    self.gamma ** self.bootstrap_steps) * next_q_value * (
                                     1 - batch['done'].view(-1, 1))

        # update
        self.optim.zero_grad()
        loss = torch.mean(0.5 * (q_value - target_q_value) ** 2)
        self.writer.add_scalar('loss/value',loss,self.loss_step)
        loss.backward()
        self.optim.step()
        self.loss_step += 1

        priority = (np.abs((q_value - target_q_value).detach().cpu().numpy()).reshape(
            -1) + self.priority_epsilon) ** self.alpha
        self.replay_memory.update_priority(index[self.burn_in_length:].reshape(-1), priority)
        self.replay_memory.update_sequence_priority(seq_index, True)

    def interval(self):
        if self.n_epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        if self.n_epochs % self.net_save_interval == 0:
            self.save_model()
        if self.n_epochs % self.memory_load_interval == 0:
            for i in range(self.n_actors):
                try:
                    self.replay_memory.load(self.memory_path, i)
                except:
                    pass
        if self.n_epochs % 10000 == 0:  # save model
            save_path = 'save/' + str(self.n_epochs) + '_save.pt'
            save_model = deepcopy(self.net).cpu().state_dict()
            torch.save(save_model, save_path)

    def save_model(self):
        self.shared_dict['net_state'] = deepcopy(self.net).cpu().state_dict()
        self.shared_dict['target_net_state'] = deepcopy(self.target_net).cpu().state_dict()
