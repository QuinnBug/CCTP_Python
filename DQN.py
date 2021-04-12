import math
import random
import torch as pt
import torch.nn as ptnn
import torch.nn.functional as ptnnf
import torch.optim as pto
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'pass_through', 'action', 'next_state', 'reward'))

STRIDE = 1
KERNEL = 5
CAM_COUNT = 4


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
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()

        self.fc1 = ptnn.Linear(inputs, 64)
        self.fc2 = ptnn.Linear(64, 32)
        self.fc3 = ptnn.Linear(32, outputs)

    def forward(self, x):
        # print("forward")
        # print(x)

        x = ptnnf.relu(self.fc1(x.view(x.size(0), -1)))
        x = ptnnf.relu(self.fc2(x.view(x.size(0), -1)))
        x = self.fc3(x.view(x.size(0), -1))

        # print(x)

        return x


class UnitNN(ptnn.Module):
    def __init__(self, h, w, outputs):
        super(UnitNN, self).__init__()
        self.conv1 = ptnn.Conv2d(3, 8, kernel_size=KERNEL, stride=STRIDE)
        self.bn1 = ptnn.BatchNorm2d(8)
        self.conv2 = ptnn.Conv2d(8, 16, kernel_size=KERNEL, stride=STRIDE)
        self.bn2 = ptnn.BatchNorm2d(16)
        self.conv3 = ptnn.Conv2d(16, 32, kernel_size=KERNEL, stride=STRIDE)
        self.bn3 = ptnn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=KERNEL, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc1 = ptnn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = ptnnf.relu(self.bn1(self.conv1(x)))
        x = ptnnf.relu(self.bn2(self.conv2(x)))
        x = ptnnf.relu(self.bn3(self.conv3(x)))
        x = self.fc1(x.view(x.size(0), -1))

        return x


class Agent:
    def __init__(self, device, n_actions, eps_start, eps_min, eps_dec, screen_size, network_runner, load_path="NONE"):
        self.load = load_path
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

        self.unit_net = [UnitNN(screen_height, screen_width, self.n_actions).to(device),
                         UnitNN(screen_height, screen_width, self.n_actions).to(device),
                         UnitNN(screen_height, screen_width, self.n_actions).to(device),
                         UnitNN(screen_height, screen_width, self.n_actions).to(device)]

        self.target_unit_net = [UnitNN(screen_height, screen_width, self.n_actions).to(device),
                                UnitNN(screen_height, screen_width, self.n_actions).to(device),
                                UnitNN(screen_height, screen_width, self.n_actions).to(device),
                                UnitNN(screen_height, screen_width, self.n_actions).to(device)]

        self.policy_net = DQN(self.n_actions, self.n_actions).to(device)
        self.target_net = DQN(self.n_actions, self.n_actions).to(device)

        self.optimizer = pto.RMSprop(self.policy_net.parameters())
        self.unit_optimizer = pto.RMSprop(self.unit_net.parameters())
        self.memory = ReplayMemory(10000)

        self.episode_durations = []
        self.episode_scores = []
        self.score_means = []

        self.steps_done = 0

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.load != "NONE":
            self.load_models(self.load)

    def save_models(self, path):
        pt.save({
            'model_state_dict_p': self.policy_net.state_dict(),
            'model_state_dict_t': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_scores': self.episode_scores,
            'episode_durations': self.episode_durations,
            'memory': self.memory
        }, path)

    def load_models(self, path):
        checkpoint = pt.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict_p'])
        self.target_net.load_state_dict(checkpoint['model_state_dict_t'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_scores = checkpoint['episode_scores']
        self.episode_durations = checkpoint['episode_durations']
        self.memory = checkpoint['memory']

        self.policy_net.eval()
        self.target_net.eval()

    def action_processing(self, state):
        sample = random.random()
        eps_threshold = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-1. * self.steps_done / self.eps_dec)

        if self.nr.receiver.game_cntr % 100 == 0:
            sample += 1

        # sample += 1

        if sample > eps_threshold:
            print("best action ap")
            with pt.no_grad():
                outputs = [self.unit_net[0](state[0]), self.unit_net[1](state[1]), self.unit_net[2](state[2]),
                           self.unit_net[3](state[3])]

                outputs = pt.stack(outputs)
        else:
            print("random action ap")
            outputs = pt.rand(4, 4)
            outputs.float()

        return outputs

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-1. * self.steps_done / self.eps_dec)
        self.steps_done += 1

        if self.nr.receiver.game_cntr % 100 == 0:
            sample += 1

        # sample += 1

        if sample > eps_threshold:
            print("best action sa")
            with pt.no_grad():
                x = self.policy_net(state).max(1)[1].view(1, 4)
                return x
        else:
            print("random action sa")
            return pt.tensor([[random.randrange(self.n_actions), random.randrange(self.n_actions),
                               random.randrange(self.n_actions), random.randrange(self.n_actions)]],
                             device=self.device, dtype=pt.long)
