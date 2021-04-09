import socket as skt
import torch as pt
from NetworkRunner import NetworkRunner
from PIL import Image
import io
import random

HOST = '127.0.0.1'
PORT = 65432
SIZE = 800000
MODEL_PATH = "D:/Documents/Coding/CCTP_Model/model_2"


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
        self.image = Image.open("BlackScreen_128.png")
        self.images = [self.image, self.image, self.image, self.image, self.image]
        self.networkRunner = NetworkRunner(self)
        self.action = pt.tensor([[0, 0, 0, 0, 0]])

        # comment out the next line to start a new model with the model path file name
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

                            img_data = data[3:].split(b'\x89PNG')
                            img_data[0] = b'\x89PNG' + img_data[1]
                            img_data[1] = b'\x89PNG' + img_data[2]
                            img_data[2] = b'\x89PNG' + img_data[3]
                            img_data[3] = b'\x89PNG' + img_data[4]
                            img_data[4] = b'\x89PNG' + img_data[5]

                            # self.data = data[3:]
                            self.cumulative_reward += self.reward

                            if data[3] == 0x89:
                                # self.image = Image.open(io.BytesIO(self.data))
                                self.images[0] = Image.open(io.BytesIO(img_data[0]))
                                self.images[1] = Image.open(io.BytesIO(img_data[1]))
                                self.images[2] = Image.open(io.BytesIO(img_data[2]))
                                self.images[3] = Image.open(io.BytesIO(img_data[3]))
                                self.images[4] = Image.open(io.BytesIO(img_data[4]))

                                self.networkRunner.run()

                                print(self.action)

                                ba = self.action.numpy().tobytes()
                                conn.sendall(ba)
                                print("completed episode/game: ")
                                print(self.networkRunner.episode_cntr)
                                print(self.game_cntr)
                                # print("action:")
                                # print(ba)
                                # print("reward = ")
                                # print(self.reward)
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
