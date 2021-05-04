from DataReciever import ImageReceiver

receiver = ImageReceiver("test")
while True:
    if receiver.game_cntr <= 100:
        receiver.update()
    else:
        receiver.networkRunner.plot_losses()
        receiver.networkRunner.plot_graphs()
