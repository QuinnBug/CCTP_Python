import numpy

from DQN import Agent
import torch as pt
import torch.nn.functional as ptnnf
import torchvision.transforms as tv
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

# import queue

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 25
SCREEN_SIZE = 16
ACTION_COUNT = 4

resize = tv.Compose([tv.Resize(SCREEN_SIZE, interpolation=Image.CUBIC),
                     tv.ToTensor()])

Transition = namedtuple('Transition',
                        ('state', 'pass_through', 'action', 'next_pass',
                         'next_state', 'reward', 'position'))

Overview = namedtuple('Overview',
                      ('image', 'processed_actions', 'reward'))


class NetworkRunner:

    def __init__(self, receiver):
        self.is_ipython = 'inline' in plt.get_backend()
        plt.ion()

        self.receiver = receiver
        self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
        self.agent = Agent(self.device, n_actions=ACTION_COUNT, eps_start=EPS_START, eps_dec=EPS_DECAY, eps_min=EPS_END,
                           screen_size=SCREEN_SIZE, network_runner=self)

        # Initialize the environment and state
        self.ov_screen = self.receiver.images[4]
        self.current_screen = self.get_screen()

        self.state = pt.stack(self.current_screen).squeeze(1)
        self.episode_cntr = 1

        self.reward = pt.tensor([0], device=self.device)
        self.done = False

        # self.previous_state = self.state
        # self.previous_action = pt.tensor([0])
        self.losses = []
        self.unit_losses = [[], [], [], []]

        self.pass_through = Overview(processed_actions=0, image=0, reward=[0, 0, 0, 0])

    def run(self):
        self.reward = pt.tensor([self.receiver.reward], device=self.device)
        self.done = self.receiver.game_over

        # Observe new state
        self.current_screen = self.get_screen()

        self.plot_state(self.current_screen[4], 6, "overview")
        # self.plot_state(self.current_screen[0], 7, "current 1")
        # self.plot_state(self.current_screen[1], 8, "current 2")
        # self.plot_state(self.current_screen[2], 9, "current 3")
        # self.plot_state(self.current_screen[3], 10, "current 4")

        if not self.done:
            next_state = pt.stack(self.current_screen).squeeze(1)
            next_pass = Overview(processed_actions=0,
                                 image=0, reward=self.receiver.rewards)
        else:
            next_state = None
            next_pass = Overview(processed_actions=0,
                                 image=0, reward=self.receiver.rewards)

        # Store the transition in memory
        self.agent.memory.push(self.state, self.pass_through, self.receiver.action,
                               next_pass, next_state, self.reward)

        # Move to the next state
        self.state = next_state
        self.pass_through = next_pass
        print("reward / action")
        print(self.pass_through.reward)
        print(self.receiver.action)

        # Select an action to send to the env
        if not self.done:
            # self.receiver.action = self.agent.select_action(self.pass_through)
            self.receiver.action = self.agent.select_action(self.state)
        else:
            print("done")
            self.optimize_model()
            self.agent.episode_durations.append(self.episode_cntr)
            self.agent.episode_scores.append(self.receiver.cumulative_reward)
            self.receiver.image = Image.open("BlackScreen_128.png")

            if self.receiver.game_cntr % 10 == 0:
                self.plot_graphs()
                self.plot_losses()

            self.episode_cntr = 0
            self.current_screen = self.get_screen()
            self.state = pt.stack(self.current_screen).squeeze(1)
            self.pass_through = Overview(processed_actions=pt.zeros(4, 1, 4), image=self.ov_screen, reward=[0, 0, 0, 0])
            self.receiver.reward = 0
            self.reward = pt.tensor([0], device=self.device)
            self.done = False
            return

        # Update the target network, copying all weights and biases in the networks
        if self.episode_cntr % TARGET_UPDATE == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # for i in range(4):
            #   self.agent.target_unit_net[i].load_state_dict(self.agent.unit_net[i].state_dict())

        self.episode_cntr += 1

    def optimize_model(self):
        if len(self.agent.memory) < BATCH_SIZE:
            return
        transitions = self.agent.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=self.agent.device, dtype=pt.bool)
        non_final_next_states = pt.cat([s for s in batch.next_state if s is not None])
        state_batch = pt.cat(batch.state)
        action_batch = pt.stack(batch.action)
        print("batches")
        x = []
        for j in range(BATCH_SIZE):
            x.append(pt.tensor(batch.pass_through[j].reward))

        reward_batch = pt.stack(x)
        # print(reward_batch.shape)
        # print(action_batch.shape)
        # print(state_batch.shape)

        state_action_values = self.agent.policy_net(state_batch).max(1)[0].gather(1, action_batch)

        next_state_values = pt.zeros((BATCH_SIZE, 4), device=self.agent.device)

        print(self.agent.target_net(non_final_next_states))
        next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1)[0]
        print(next_state_values)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = ptnnf.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.agent.optimizer.zero_grad()
        print("loss")
        print(loss)
        loss.backward()
        self.losses.append(loss)

        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp(-1, 1)
        self.agent.optimizer.step()

    def get_screen(self):
        x = []

        screen = self.receiver.images[0]
        x.append(resize(screen).unsqueeze(0))
        screen = self.receiver.images[1]
        x.append(resize(screen).unsqueeze(0))
        screen = self.receiver.images[2]
        x.append(resize(screen).unsqueeze(0))
        screen = self.receiver.images[3]
        x.append(resize(screen).unsqueeze(0))

        screen = self.receiver.images[4]
        # self.ov_screen = resize(screen).unsqueeze(0)
        x.append(resize(screen).unsqueeze(0))

        return x

    def plot_state(self, state, figure=5, name="state"):
        plt.figure(figure)
        plt.clf()
        plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        plt.title(name)
        plt.pause(0.001)
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_losses(self):

        losses_t = pt.tensor(self.losses, dtype=pt.float)
        if len(losses_t) < 1:
            return

        plt.figure(3)
        plt.clf()
        plt.title('Fully Connected Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

        # losses_t = [pt.tensor(self.unit_losses[0], dtype=pt.float), pt.tensor(self.unit_losses[1], dtype=pt.float),
        #             pt.tensor(self.unit_losses[2], dtype=pt.float), pt.tensor(self.unit_losses[3], dtype=pt.float)]
        # if len(losses_t) < 1:
        #     return
        #
        # plt.figure(4)
        # plt.clf()
        # plt.title('Convolutional NN Loss')
        # plt.xlabel('Steps')
        # plt.ylabel('Loss')
        # plt.plot(losses_t[0].numpy())
        # plt.plot(losses_t[1].numpy())
        # plt.plot(losses_t[2].numpy())
        # plt.plot(losses_t[3].numpy())
        #
        # plt.pause(0.001)  # pause a bit so that plots are updated
        # if self.is_ipython:
        #     display.clear_output(wait=True)
        #     display.display(plt.gcf())

    def plot_graphs(self):
        plt.figure(2)
        plt.clf()
        # durations_t = pt.tensor(self.agent.episode_durations, dtype=pt.float)
        scores_t = pt.tensor(self.agent.episode_scores, dtype=pt.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration & Score')
        # plt.plot(durations_t.numpy())
        plt.plot(scores_t.numpy())

        # if len(durations_t) >= 100:
        #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        #     means = pt.cat((pt.zeros(99), means))
        #     plt.plot(means.numpy())

        if len(scores_t) % 100 == 0 & len(scores_t) >= 100:
            hundo_count = len(scores_t) / 100
            i = 0
            j = 0
            self.agent.score_means = []
            while i <= hundo_count:
                while j <= 100:
                    self.agent.score_means.append(np.sum(self.agent.episode_scores[i * 100, (i * 100) + 101]))
                    j += 1
                i += 1

        if len(scores_t) >= 100:
            plt.plot(self.agent.score_means)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
