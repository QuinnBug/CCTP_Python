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
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 50
SCREEN_SIZE = 16
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
        # self.last_screen = self.get_screen()
        self.current_screen = self.get_screen()
        self.blended_img = self.current_screen
        self.last_screens = [self.current_screen, self.current_screen, self.current_screen]

        self.state = self.blend_screens()
        self.episode_cntr = 1

        self.reward = pt.tensor([0], device=self.device)
        self.done = False

        self.previous_state = self.state
        self.previous_action = pt.tensor([[0, 0, 0, 0]])
        self.losses = []
        self.avg_losses = []

    def blend_screens(self):
        # image1 = tv.ToPILImage()(np.squeeze(self.current_screen))
        # image2 = tv.ToPILImage()(np.squeeze(self.last_screens[0]))
        # image3 = tv.ToPILImage()(np.squeeze(self.last_screens[1]))
        # image4 = tv.ToPILImage()(np.squeeze(self.last_screens[2]))

        # self.blended_img = Image.blend(image1, image2, alpha=0.05)
        # self.blended_img = Image.blend(self.blended_img, image3, alpha=0.03)
        # self.blended_img = Image.blend(self.blended_img, image4, alpha=0.01).convert("RGB")

        self.blended_img = self.current_screen

        return self.blended_img
        # return resize(self.blended_img).unsqueeze(0).to(self.device)

    def run(self):
        self.reward = pt.tensor([self.receiver.rewards], device=self.device)
        self.done = self.receiver.game_over

        # Observe new state
        self.current_screen = self.get_screen()

        # self.plot_state(self.current_screen, name="current", figure=6)

        if not self.done:
            next_state = self.blend_screens()
        else:
            next_state = None

        # Store the transition in memory
        self.agent.memory.push(self.state, self.receiver.action, next_state, self.reward)

        # Move to the next state
        self.state = next_state

        # Select an action to send to the env
        if self.state is not None:
            self.receiver.action = self.agent.select_action(self.state)

        # Perform one step of the optimization (on the target network)
        self.optimize_model()
        self.plot_graphs()
        self.plot_losses()

        if self.done:
            print("done")
            self.agent.episode_durations.append(self.episode_cntr)
            self.agent.episode_scores.append(self.receiver.cumulative_reward)
            self.receiver.image = Image.open("BlackScreen_128.png")

            self.episode_cntr = 0
            self.current_screen = self.get_screen()
            self.last_screens = [self.current_screen, self.current_screen, self.current_screen]
            self.state = self.blend_screens()
            self.receiver.reward = 0
            self.reward = pt.tensor([0], device=self.device)
            self.done = False
            return

        # Update the target network, copying all weights and biases in DQN
        if self.episode_cntr % TARGET_UPDATE == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

        self.episode_cntr += 1

        # self.plot_state(self.state, name="state")

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

        next_state_values = pt.zeros((BATCH_SIZE, 4), device=self.agent.device)

        o = self.agent.target_net(non_final_next_states).max(1)[0]
        o = o.reshape(int(o.shape[0]/4), 4)
        next_state_values[non_final_mask] = o.detach()

        # print("debug")
        # print(next_state_values[non_final_mask])

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = ptnnf.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.agent.optimizer.zero_grad()

        print("loss")
        print(loss)
        loss.backward()
        print(loss)

        self.losses.append(loss)
        t = 0
        for i in range(len(self.losses)):
            t += self.losses[i]

        t = t / len(self.losses)
        self.avg_losses.append(t)

        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()

    def get_screen(self):
        x = []

        # think about a for loop? not super needed but would be useful

        screen = self.receiver.images[0]
        x.append(resize(screen))

        screen = self.receiver.images[1]
        x.append(resize(screen))

        screen = self.receiver.images[2]
        x.append(resize(screen))

        screen = self.receiver.images[3]
        x.append(resize(screen))

        screen = self.receiver.images[4]
        x.append(resize(screen))

        y = [pt.cat([x[0], x[4]], dim=1), pt.cat([x[1], x[4]], dim=1),
             pt.cat([x[2], x[4]], dim=1), pt.cat([x[3], x[4]], dim=1)]

        self.plot_state(pt.cat([pt.cat([y[0], y[1]], dim=2), pt.cat([y[2], y[3]], dim=2)],
                               dim=1), name="current 1", figure=6)

        y = pt.stack(y)
        return y

    def plot_state(self, state, figure=4, name="state"):
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
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('loss')
        plt.plot(losses_t.numpy())
        plt.plot(avg_t.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

        avg_t = pt.tensor(self.avg_losses, dtype=pt.float)

        plt.plot(avg_t.numpy())

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
