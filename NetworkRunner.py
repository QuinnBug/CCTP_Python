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

BATCH_SIZE = 2048
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
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

        self.state = self.current_screen
        self.episode_cntr = 1

        self.reward = pt.tensor([0], device=self.device)
        self.done = False

        self.previous_state = self.state
        self.previous_action = pt.tensor([0])
        self.losses = []
        self.avg_losses = []
        self.unit_losses = [[], [], [], []]
        self.avg_unit_losses = [[], [], [], []]

        self.pass_through = Overview(processed_actions=pt.stack(self.agent.action_processing(self.state)),
                                     image=self.ov_screen, reward=[0, 0, 0, 0])

    def run(self):
        self.reward = pt.tensor([self.receiver.reward], device=self.device)
        self.done = self.receiver.game_over

        # Observe new state
        self.current_screen = self.get_screen()

        self.plot_state(self.ov_screen, 6, "overview")

        if not self.done:
            next_state = self.current_screen
            next_pass = Overview(processed_actions=pt.stack(self.agent.action_processing(next_state)),
                                 image=self.ov_screen, reward=self.receiver.rewards)
        else:
            next_state = [None, None, None, None]
            next_pass = Overview(processed_actions=pt.zeros(4, 1, 4),
                                 image=self.ov_screen, reward=self.receiver.rewards)

        # Store the transition in memory
        self.agent.memory.push(self.state, self.pass_through, self.receiver.action, next_pass, next_state, self.reward)

        # Move to the next state
        self.state = next_state
        self.pass_through = next_pass

        # Select an action to send to the env
        if not self.done:
            self.receiver.action = self.agent.select_action(self.pass_through)
        else:
            print("done")
            self.optimize_model()
            self.plot_graphs()
            self.plot_losses()
            self.agent.episode_durations.append(self.episode_cntr)
            self.agent.episode_scores.append(self.receiver.cumulative_reward)
            self.receiver.image = Image.open("BlackScreen_128.png")
            self.episode_cntr = 0
            self.current_screen = self.get_screen()
            self.state = self.current_screen
            self.pass_through = Overview(processed_actions=pt.zeros(4, 1, 4), image=self.ov_screen, reward=[0, 0, 0, 0])
            self.receiver.reward = 0
            self.reward = pt.tensor([0], device=self.device)
            self.done = False
            return

        # Update the target network, copying all weights and biases in the networks
        if self.episode_cntr % TARGET_UPDATE == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            for i in range(4):
                self.agent.target_unit_net[i].load_state_dict(self.agent.unit_net[i].state_dict())

        self.episode_cntr += 1

    def optimize_model(self):
        if len(self.agent.memory) < BATCH_SIZE:
            return
        transitions = self.agent.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=self.agent.device, dtype=pt.bool)

        x = []
        y = []
        for i in range(BATCH_SIZE):
            y.append(batch.position[i])
            if batch.next_pass[i].processed_actions[0] is not None:
                x.append(batch.position[i])

        non_final_next_passes = pt.cat([pt.ones(1), pt.tensor(x)]).unsqueeze(1)
        pass_batch = pt.cat([pt.zeros(1), pt.tensor(y)]).unsqueeze(1)

        action_batch = pt.cat(batch.action).unsqueeze(1)
        reward_batch = pt.cat(batch.reward)

        pass_action_values = self.agent.policy_net(pass_batch).gather(1, action_batch)
        next_pass_values = pt.zeros((BATCH_SIZE, 4), device=self.agent.device)
        next_pass_values[non_final_mask] = self.agent.target_net(non_final_next_passes).max(1)[0][0].detach()

        x = []
        for j in range(BATCH_SIZE):
            x.append(pt.tensor([batch.pass_through[j].reward[0], batch.pass_through[j].reward[1],
                               batch.pass_through[j].reward[2], batch.pass_through[j].reward[3]]))

        rb_ext = reward_batch.unsqueeze(1)
        rb_ext = rb_ext.repeat(1, 4)

        print("test")
        print(reward_batch.shape)

        reward_batch = pt.stack(x)

        print(rb_ext.shape)
        print(reward_batch.shape)

        # expected_pass_action_values = (next_pass_values * GAMMA) + rb_ext
        expected_pass_action_values = (next_pass_values * GAMMA) + reward_batch

        loss = ptnnf.smooth_l1_loss(pass_action_values, expected_pass_action_values.unsqueeze(1))
        self.agent.optimizer.zero_grad()

        loss.backward()

        self.losses.append(loss)

        t = 0
        for i in range(len(self.losses)):
            t += self.losses[i]

        t = t / len(self.losses)
        self.avg_losses.append(t)

        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()

        self.optimize_unit_models()

    def optimize_unit_models(self):
        if len(self.agent.memory) < BATCH_SIZE:
            return

        transitions = self.agent.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=self.agent.device, dtype=pt.bool)

        for i in range(4):
            x = []
            for state in batch.next_state:
                if state[0] is not None:
                    x.append(state[i])
                else:
                    x.append(pt.zeros(1, 3, SCREEN_SIZE, SCREEN_SIZE))
            non_final_next_states = pt.cat(x)

            x = []
            for state in batch.state:
                if state[0] is not None:
                    x.append(state[i])
                else:
                    x.append(pt.zeros(1, 3, SCREEN_SIZE, SCREEN_SIZE))
            state_batch = pt.cat(x)

            y = []
            x = []
            for j in range(BATCH_SIZE):
                x.append(pt.tensor(batch.pass_through[j].reward[i]))
                y.append(batch.pass_through[j].processed_actions.max(0)[1])

            z = []
            for k in range(BATCH_SIZE):
                z.append(y[k][0][i])

            pass_batch = pt.stack(z)
            pass_batch = pass_batch.unsqueeze(1).repeat(1, 4)
            reward_batch = pt.stack(x)

            state_pass_values = self.agent.unit_net[i](state_batch).gather(0, pass_batch)
            next_state_values = pt.zeros(BATCH_SIZE, device=self.agent.device)
            next_state_values[non_final_mask] = self.agent.target_unit_net[i](non_final_next_states).max(1)[0].detach()
            expected_state_pass_values = (next_state_values * GAMMA) + reward_batch

            expected_state_pass_values = expected_state_pass_values.unsqueeze(1).repeat(1, 4)

            unit_loss = ptnnf.smooth_l1_loss(state_pass_values, expected_state_pass_values)
            self.agent.unit_optimizer[i].zero_grad()

            unit_loss.backward()

            self.unit_losses[i].append(unit_loss)
            t = 0
            for j in range(len(self.unit_losses[i])):
                t += self.unit_losses[i][j]

            t = t / len(self.unit_losses[i])
            self.avg_unit_losses[i].append(t)

            for param in self.agent.unit_net[i].parameters():
                param.grad.data.clamp_(-1, 1)
            self.agent.unit_optimizer[i].step()

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
        self.ov_screen = resize(screen).unsqueeze(0)

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
        if len(self.losses) < 1:
            return

        losses_t = pt.tensor(self.losses, dtype=pt.float)
        avg_t = pt.tensor(self.avg_losses, dtype=pt.float)

        plt.figure(3)
        plt.clf()
        plt.title('Fully Connected Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())
        plt.plot(avg_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

        losses_t = [pt.tensor(self.unit_losses[0], dtype=pt.float), pt.tensor(self.unit_losses[1], dtype=pt.float),
                    pt.tensor(self.unit_losses[2], dtype=pt.float), pt.tensor(self.unit_losses[3], dtype=pt.float)]
        avg_t = [pt.tensor(self.avg_unit_losses[0], dtype=pt.float), pt.tensor(self.avg_unit_losses[1], dtype=pt.float),
                 pt.tensor(self.avg_unit_losses[2], dtype=pt.float), pt.tensor(self.avg_unit_losses[3], dtype=pt.float)]
        if len(losses_t) < 1:
            return

        plt.figure(4)
        plt.clf()
        plt.title('Convolutional NN Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.plot(losses_t[0].numpy())
        plt.plot(losses_t[1].numpy())
        plt.plot(losses_t[2].numpy())
        plt.plot(losses_t[3].numpy())
        plt.plot(avg_t[0].numpy())
        plt.plot(avg_t[1].numpy())
        plt.plot(avg_t[2].numpy())
        plt.plot(avg_t[3].numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_graphs(self):
        plt.figure(2)
        plt.clf()
        scores_t = pt.tensor(self.agent.episode_scores, dtype=pt.float)
        plt.title('Training')
        plt.xlabel('Games')
        plt.ylabel('Score')
        plt.plot(scores_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
