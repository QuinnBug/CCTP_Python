import math
import random
import torch as pt
import torch.nn as ptnn
import torch.nn.functional as ptnnf
import torch.optim as pto
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(ptnn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = ptnn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = ptnn.BatchNorm2d(16)
        self.conv2 = ptnn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = ptnn.BatchNorm2d(32)
        self.conv3 = ptnn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = ptnn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = ptnn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = ptnnf.relu(self.bn1(self.conv1(x)))
        x = ptnnf.relu(self.bn2(self.conv2(x)))
        x = ptnnf.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent:
    def __init__(self, device, n_actions, eps_start, eps_min, eps_dec, screen_size, network_runner):
        self.device = device

        # Set self.params
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.screen_size = screen_size

        self.nr = network_runner
        init_screen = self.nr.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = pto.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.episode_durations = []
        self.episode_scores = []

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-1. * self.steps_done / self.eps_dec)
        self.steps_done += 1
        if sample > eps_threshold:
            print("best action - eps current/sample")
            with pt.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            print("random action - eps current/sample")
            return pt.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=pt.long)
