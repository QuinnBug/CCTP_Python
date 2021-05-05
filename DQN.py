import math
import random
import torch as pt
import torch.nn as ptnn
import torch.nn.functional as ptnnf
import torch.optim as pto
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

STRIDE = 1
KERNEL = 6
CAM_COUNT = 5


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
        self.conv1 = ptnn.Conv2d(3, 16, kernel_size=KERNEL, stride=STRIDE)
        self.bn1 = ptnn.BatchNorm2d(16)
        self.conv2 = ptnn.Conv2d(16, 32, kernel_size=KERNEL, stride=STRIDE)
        self.bn2 = ptnn.BatchNorm2d(32)
        self.conv3 = ptnn.Conv2d(32, 16, kernel_size=KERNEL, stride=STRIDE)
        self.bn3 = ptnn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=KERNEL, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16

        self.fc1 = ptnn.Linear(linear_input_size, 24)
        self.fc2 = ptnn.Linear(24, outputs)

    def forward(self, x):
        x = ptnnf.relu(self.bn1(self.conv1(x)))
        x = ptnnf.relu(self.bn2(self.conv2(x)))
        x = ptnnf.relu(self.bn3(self.conv3(x)))
        x = ptnnf.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x.view(x.size(0), -1))

        return x


class Agent:
    def __init__(self, device, n_actions, eps_start, eps_min, eps_dec, screen_size, network_runner, load_path="NONE"):
        self.load = load_path
        self.device = device

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
        self.optimizer = pto.RMSprop(self.policy_net.parameters(), lr=1e-04)
        self.memory = ReplayMemory(1000000)

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

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-1. * self.steps_done / self.eps_dec)
        self.steps_done += 1

        if self.nr.receiver.game_cntr % 25 == 0:
            print("BEST TEST")
            sample = 1

        # sample += 1

        if sample > eps_threshold:
            print("best action")
            with pt.no_grad():
                x = self.policy_net(state)
                return x.max(1)[1].view(1, 4)
        else:
            print("random action")
            return pt.tensor([[random.randrange(self.n_actions), random.randrange(self.n_actions),
                              random.randrange(self.n_actions), random.randrange(self.n_actions)]],
                             device=self.device, dtype=pt.long)
