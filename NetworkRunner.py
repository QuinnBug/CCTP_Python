from DQN import Agent
import torch as pt
import torch.nn.functional as ptnnf
import torchvision.transforms as tv
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 1000
TARGET_UPDATE = 25
SCREEN_SIZE = 128
ACTION_COUNT = 3

resize = tv.Compose([tv.Resize(SCREEN_SIZE, interpolation=Image.CUBIC),
                     tv.ToTensor()])

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class NetworkRunner:

    def __init__(self, receiver):
        self.is_ipython = 'inline' in plt.get_backend()
        plt.ion()

        self.receiver = receiver
        self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
        self.agent = Agent(self.device, n_actions=ACTION_COUNT, eps_start=EPS_START, eps_dec=EPS_DECAY, eps_min=EPS_END,
                           screen_size=SCREEN_SIZE, network_runner=self)

        # Initialize the environment and state
        self.last_screen = self.get_screen()
        self.current_screen = self.get_screen()
        # self.state = self.current_screen - self.last_screen
        self.state = self.current_screen
        self.episode_cntr = 1

        self.reward = pt.tensor([0], device=self.device)
        self.done = False

        self.previous_state = self.state
        self.previous_action = pt.tensor([[0]])

    def run(self):

        self.reward = pt.tensor([self.receiver.reward], device=self.device)
        self.done = self.receiver.game_over

        # Observe new state
        self.last_screen = self.current_screen
        self.current_screen = self.get_screen()

        if not self.done:
            next_state = self.current_screen
        else:
            self.receiver.image = Image.open("BlackScreen.png")
            next_state = None

            # Store the transition in memory
            # self.agent.memory.push(self.state, self.receiver.action, next_state, self.reward)
            self.agent.memory.push(self.previous_state, self.previous_action, self.state, self.reward)

        # Select and perform an action
        self.previous_action = self.receiver.action
        self.receiver.action = self.agent.select_action(self.previous_state)
        print(self.receiver.action)

        # Move to the next state
        self.previous_state = self.state
        self.state = next_state

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        if self.done:
            print("done")
            self.agent.episode_durations.append(self.episode_cntr)
            self.agent.episode_scores.append(self.receiver.cumulative_reward)
            # if self.receiver.game_cntr % 25 == 0:
            #     self.plot_graphs()

            self.episode_cntr = 0
            self.current_screen = self.get_screen()
            self.last_screen = self.current_screen
            self.state = self.current_screen - self.last_screen
            self.receiver.reward = 0
            self.reward = pt.tensor([0], device=self.device)
            self.done = False
            return

        # Update the target network, copying all weights and biases in DQN
        if self.episode_cntr % TARGET_UPDATE == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

        self.episode_cntr += 1

        self.plot_state(self.previous_state, figure=3, name="previous")
        # self.plot_state(self.state, name="current")

    def optimize_model(self):
        if len(self.agent.memory) < BATCH_SIZE:
            return
        transitions = self.agent.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=self.agent.device, dtype=pt.bool)
        non_final_next_states = pt.cat([s for s in batch.next_state if s is not None])
        state_batch = pt.cat(batch.state)
        action_batch = pt.cat(batch.action)
        reward_batch = pt.cat(batch.reward)

        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        next_state_values = pt.zeros(BATCH_SIZE, device=self.agent.device)
        next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = ptnnf.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.agent.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()

    def get_screen(self):
        screen = self.receiver.image
        return resize(screen).unsqueeze(0).to(self.device)

    def plot_state(self, state, figure=1, name="state"):
        plt.figure(figure)
        plt.clf()
        plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        plt.title(name)
        plt.pause(0.001)
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

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
                    self.agent.score_means.append(np.sum(self.agent.episode_scores[i*100, (i*100) + 101]))
                    j += 1
                i += 1

        if len(scores_t) >= 100:
            plt.plot(self.agent.score_means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
