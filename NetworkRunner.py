import numpy as np
from DQN import Agent
import torch as pt
import torch.nn.functional as ptnnf
import torchvision.transforms as tv
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 50
SCREEN_SIZE = 128
ACTION_COUNT = 4

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
        self.state = self.current_screen - self.last_screen
        self.episode_cntr = 1

    def run(self):
        reward = pt.tensor([self.receiver.reward], device=self.device)
        done = self.receiver.game_over

        # Observe new state
        self.last_screen = self.current_screen
        self.current_screen = self.get_screen()

        if not done:
            next_state = self.current_screen - self.last_screen
        else:
            self.receiver.image = Image.open("BlackScreen.png")
            next_state = None

        # Store the transition in memory
        self.agent.memory.push(self.state, self.receiver.action, next_state, reward)

        # Select and perform an action
        self.receiver.action = self.agent.select_action(self.state)

        # Move to the next state
        self.state = next_state

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        if done:
            print("done")
            self.agent.episode_durations.append(self.episode_cntr)
            self.agent.episode_scores.append(self.receiver.cumulative_reward)
            self.plot_graphs()

            self.episode_cntr = 0
            self.current_screen = self.get_screen()
            self.last_screen = self.current_screen
            self.state = self.current_screen - self.last_screen
            return

        # Update the target network, copying all weights and biases in DQN
        if self.episode_cntr % TARGET_UPDATE == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

        self.episode_cntr += 1

        self.plot_state()
        # plt.figure()
        # plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        #            interpolation='none')
        # plt.title('Example extracted screen')
        # plt.show()

    def optimize_model(self):
        if len(self.agent.memory) < BATCH_SIZE:
            return
        transitions = self.agent.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=self.agent.device, dtype=pt.bool)
        non_final_next_states = pt.cat([s for s in batch.next_state if s is not None])
        state_batch = pt.cat(batch.state)
        action_batch = pt.cat(batch.action)
        reward_batch = pt.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = pt.zeros(BATCH_SIZE, device=self.agent.device)
        next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = ptnnf.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.agent.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()

    def get_screen(self):
        # need to convert the image received from unity into a numpy array and transpose it
        screen = self.receiver.image
        return resize(screen).unsqueeze(0).to(self.device)

    def plot_state(self):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
        plt.title('current input screen')
        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_graphs(self):
        plt.figure(2)
        plt.clf()
        durations_t = pt.tensor(self.agent.episode_durations, dtype=pt.float)
        scores_t = pt.tensor(self.agent.episode_scores, dtype=pt.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration & Score')
        plt.plot(durations_t.numpy())
        plt.plot(scores_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = pt.cat((pt.zeros(99), means))
            plt.plot(means.numpy())

        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = pt.cat((pt.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
