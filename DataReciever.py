import socket as skt
import torch as pt
from NetworkRunner import NetworkRunner
from PIL import Image
import io
import random

HOST = '127.0.0.1'
PORT = 65432
SIZE = 200000
MODEL_PATH = "C:/Users/Quinn/Documents/CCTP_Python/PTModels/test_2_model"


class ImageReceiver:
    def __init__(self, name):
        self.name = name
        self.listening = 0
        self.active = True
        self.connected = False
        self.game_over = False
        self.reward = 0
        self.cumulative_reward = 0
        self.game_cntr = 0
        self.highest_score = -999
        self.data = []
        self.image = Image.open("BlackScreen.png")
        self.networkRunner = NetworkRunner(self)
        self.action = pt.tensor([[random.randrange(3)]])

        # uncomment the next line to load from last session
        # self.networkRunner.agent.load_models(MODEL_PATH)

    def update(self):
        if self.listening == 0:
            # listen for data and say we're listening for data
            self.receive()

    def save_models(self):
        self.networkRunner.agent.save_models(MODEL_PATH)

    def receive(self):
        with skt.socket(skt.AF_INET, skt.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print('Connecting...')
            conn, address = s.accept()
            with conn:
                print('Connected by', address)
                self.connected = True
                self.game_cntr += 1
                while self.active:
                    data = conn.recv(SIZE)
                    if data:
                        if len(data) < 2:
                            self.active = False
                        else:
                            self.game_over = True if int(data[0]) == 1 else False
                            self.reward = int(data[2]) if int(data[1]) == 0 else int(data[2]) * -1
                            self.data = data[3:]
                            self.cumulative_reward += self.reward

                            if data[3] == 0x89:
                                self.image = Image.open(io.BytesIO(self.data))

                                self.networkRunner.run()

                                ba = self.action.numpy().tobytes()
                                conn.sendall(ba)
                                # print("completed episode/game: ")
                                # print(self.networkRunner.episode_cntr)
                                # print(self.game_cntr)
                                print("reward = ")
                                print(self.reward)
                                # print(self.cumulative_reward)

                                if self.game_over:
                                    print("cumulative reward = ")
                                    print(self.cumulative_reward)
                                    # print("highest reward = ")
                                    # print(self.highest_score)

                                    # update the highest score
                                    if self.cumulative_reward > self.highest_score:
                                        self.highest_score = self.cumulative_reward

                                    self.cumulative_reward = 0
                                    self.active = False
                            else:
                                print("corrupted data")
                                print(data[:10])
                                conn.sendall(bytearray(200))

                self.active = True
                self.connected = False
