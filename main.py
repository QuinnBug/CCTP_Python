import numpy as np
from DQN import Agent
from DataReciever import ImageReceiver
# from NetworkRunner import

receiver = ImageReceiver("test")
while True:
    receiver.update()
