from DataReciever import ImageReceiver

receiver = ImageReceiver("test")
while True:
    receiver.update()
