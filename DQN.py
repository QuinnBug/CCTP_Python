import math
import random
import torch as pt
import torch.nn as ptnn
import torch.nn.functional as ptnnf
import torch.optim as pto
from collections import namedtuple

import NetworkRunner

Transition = namedtuple('Transition',
                        ('state', 'pass_through', 'action', 'next_pass',
                         'next_state', 'reward', 'position'))

Overview = namedtuple('Overview',
                      ('image', 'processed_actions', 'reward'))

STRIDE = 1
KERNEL = 4
CAM_COUNT = 4


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args, position=self.position)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class UnitNN(ptnn.Module):
    def __init__(self, h, w, outputs, agent):
        super(UnitNN, self).__init__()
        self.one_conv1 = ptnn.Conv2d(3, 8, kernel_size=KERNEL, stride=STRIDE)
        self.one_bn1 = ptnn.BatchNorm2d(8)
        self.one_conv2 = ptnn.Conv2d(8, 8, kernel_size=KERNEL, stride=STRIDE)
        self.one_bn2 = ptnn.BatchNorm2d(8)
        self.one_conv3 = ptnn.Conv2d(8, 16, kernel_size=KERNEL, stride=STRIDE)
        self.one_bn3 = ptnn.BatchNorm2d(16)
        # self.one_conv4 = ptnn.Conv2d(16, 16, kernel_size=KERNEL, stride=STRIDE)
        # self.one_bn4 = ptnn.BatchNorm2d(16)

        self.two_conv1 = ptnn.Conv2d(3, 8, kernel_size=KERNEL, stride=STRIDE)
        self.two_bn1 = ptnn.BatchNorm2d(8)
        self.two_conv2 = ptnn.Conv2d(8, 8, kernel_size=KERNEL, stride=STRIDE)
        self.two_bn2 = ptnn.BatchNorm2d(8)
        self.two_conv3 = ptnn.Conv2d(8, 16, kernel_size=KERNEL, stride=STRIDE)
        self.two_bn3 = ptnn.BatchNorm2d(16)
        # self.two_conv4 = ptnn.Conv2d(16, 16, kernel_size=KERNEL, stride=STRIDE)
        # self.two_bn4 = ptnn.BatchNorm2d(16)

        self.three_conv1 = ptnn.Conv2d(3, 8, kernel_size=KERNEL, stride=STRIDE)
        self.three_bn1 = ptnn.BatchNorm2d(8)
        self.three_conv2 = ptnn.Conv2d(8, 8, kernel_size=KERNEL, stride=STRIDE)
        self.three_bn2 = ptnn.BatchNorm2d(8)
        self.three_conv3 = ptnn.Conv2d(8, 16, kernel_size=KERNEL, stride=STRIDE)
        self.three_bn3 = ptnn.BatchNorm2d(16)
        # self.three_conv4 = ptnn.Conv2d(16, 16, kernel_size=KERNEL, stride=STRIDE)
        # self.three_bn4 = ptnn.BatchNorm2d(16)

        self.four_conv1 = ptnn.Conv2d(3, 8, kernel_size=KERNEL, stride=STRIDE)
        self.four_bn1 = ptnn.BatchNorm2d(8)
        self.four_conv2 = ptnn.Conv2d(8, 8, kernel_size=KERNEL, stride=STRIDE)
        self.four_bn2 = ptnn.BatchNorm2d(8)
        self.four_conv3 = ptnn.Conv2d(8, 16, kernel_size=KERNEL, stride=STRIDE)
        self.four_bn3 = ptnn.BatchNorm2d(16)
        # self.four_conv4 = ptnn.Conv2d(16, 16, kernel_size=KERNEL, stride=STRIDE)
        # self.four_bn4 = ptnn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=KERNEL, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = (convw * convh * 16) * 4

        print(h)
        print(w)

        self.fc1 = ptnn.Linear(linear_input_size, 32)
        self.fc2 = ptnn.Linear(32, 16)
        self.fc3 = ptnn.Linear(16, outputs)

        self.agent = agent

    def forward(self, x):
        if x.shape[0] is not 5:
            x = x.view(int(x.shape[0] / 5), 5, 3, 16, 16)
            z = []
            for i in range(x.shape[0]):
                img1 = pt.cat([x[i][0], x[i][4]], dim=1).unsqueeze(0)
                img2 = pt.cat([x[i][1], x[i][4]], dim=1).unsqueeze(0)
                img3 = pt.cat([x[i][2], x[i][4]], dim=1).unsqueeze(0)
                img4 = pt.cat([x[i][3], x[i][4]], dim=1).unsqueeze(0)

                # self.agent.nr.plot_state(img1, i+5, "image comb")

                img1 = ptnnf.relu(self.one_bn1(self.one_conv1(img1)))
                img1 = ptnnf.relu(self.one_bn2(self.one_conv2(img1)))
                img1 = ptnnf.relu(self.one_bn3(self.one_conv3(img1)))
                # img1 = ptnnf.relu(self.one_bn4(self.one_conv4(img1)))

                img2 = ptnnf.relu(self.two_bn1(self.two_conv1(img2)))
                img2 = ptnnf.relu(self.two_bn2(self.two_conv2(img2)))
                img2 = ptnnf.relu(self.two_bn3(self.two_conv3(img2)))
                # img2 = ptnnf.relu(self.two_bn4(self.two_conv4(img2)))

                img3 = ptnnf.relu(self.three_bn1(self.three_conv1(img3)))
                img3 = ptnnf.relu(self.three_bn2(self.three_conv2(img3)))
                img3 = ptnnf.relu(self.three_bn3(self.three_conv3(img3)))
                # img3 = ptnnf.relu(self.three_bn4(self.three_conv4(img3)))

                img4 = ptnnf.relu(self.four_bn1(self.four_conv1(img4)))
                img4 = ptnnf.relu(self.four_bn2(self.four_conv2(img4)))
                img4 = ptnnf.relu(self.four_bn3(self.four_conv3(img4)))
                # img4 = ptnnf.relu(self.four_bn4(self.four_conv4(img4)))

                y = pt.cat([img1, img2, img3, img4], dim=1)

                y = ptnnf.relu(self.fc1(y.view(y.size(0), -1)))
                y = ptnnf.relu(self.fc2(y.view(y.size(0), -1)))
                y = self.fc3(y.view(y.size(0), -1))

                z.append(y.view(4, 4))
            z = pt.stack(z)
            return z
        else:
            img1 = pt.cat([x[0], x[4]], dim=1).unsqueeze(0)
            img2 = pt.cat([x[1], x[4]], dim=1).unsqueeze(0)
            img3 = pt.cat([x[2], x[4]], dim=1).unsqueeze(0)
            img4 = pt.cat([x[3], x[4]], dim=1).unsqueeze(0)

            img1 = ptnnf.relu(self.one_bn1(self.one_conv1(img1)))
            img1 = ptnnf.relu(self.one_bn2(self.one_conv2(img1)))
            img1 = ptnnf.relu(self.one_bn3(self.one_conv3(img1)))
            # img1 = ptnnf.relu(self.one_bn4(self.one_conv4(img1)))

            img2 = ptnnf.relu(self.two_bn1(self.two_conv1(img2)))
            img2 = ptnnf.relu(self.two_bn2(self.two_conv2(img2)))
            img2 = ptnnf.relu(self.two_bn3(self.two_conv3(img2)))
            # img2 = ptnnf.relu(self.two_bn4(self.two_conv4(img2)))

            img3 = ptnnf.relu(self.three_bn1(self.three_conv1(img3)))
            img3 = ptnnf.relu(self.three_bn2(self.three_conv2(img3)))
            img3 = ptnnf.relu(self.three_bn3(self.three_conv3(img3)))
            # img3 = ptnnf.relu(self.three_bn4(self.three_conv4(img3)))

            img4 = ptnnf.relu(self.four_bn1(self.four_conv1(img4)))
            img4 = ptnnf.relu(self.four_bn2(self.four_conv2(img4)))
            img4 = ptnnf.relu(self.four_bn3(self.four_conv3(img4)))
            # img4 = ptnnf.relu(self.four_bn4(self.four_conv4(img4)))

            x = pt.cat([img1, img2, img3, img4], dim=1)

            x = ptnnf.relu(self.fc1(x.view(x.size(0), -1)))
            x = ptnnf.relu(self.fc2(x.view(x.size(0), -1)))
            x = self.fc3(x.view(x.size(0), -1))

            return x.view(4, 4)


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
        init_screen = self.nr.get_screen()[0]
        _, _, screen_height, screen_width = init_screen.shape

        self.policy_net = UnitNN(screen_height * 2, screen_width, self.n_actions * 4, self).to(device)
        self.target_net = UnitNN(screen_height * 2, screen_width, self.n_actions * 4, self).to(device)

        self.optimizer = pto.RMSprop(self.policy_net.parameters(), lr=1e-12)

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

        if self.nr.receiver.game_cntr % 100 == 0:
            sample += 1

        # sample += 1

        if sample > eps_threshold:
            with pt.no_grad():
                x = self.policy_net(state)
                return x.max(1)[1]
        else:
            # print("random action sa")
            return pt.tensor([random.randrange(self.n_actions),
                              random.randrange(self.n_actions),
                              random.randrange(self.n_actions),
                              random.randrange(self.n_actions)],
                             device=self.device, dtype=pt.long)
