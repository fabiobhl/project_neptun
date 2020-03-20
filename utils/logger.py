from progress.bar import IncrementalBar

class ProgressBar(IncrementalBar):

    def __init__(self, message, maximum):
        super().__init__()
        self.meanloss = 0
        self.message = message
        self.max = maximum
        self.suffix = '0'
        self.dash = '/'

    def step(self, meanloss):
        self.next()
        self.meanloss = meanloss
        self.suffix = f'AvgLoss: {round(self.meanloss,4)}, Batches: %(index)d{self.dash}%(max)d'

    def lastcall(self, val_accuracy, val_loss):
        self.suffix = f'AvgLoss: {round(self.meanloss,4)}  Batches: %(index)d{self.dash}%(max)d || val_accuracy: {round(val_accuracy,4)}  val_loss: {round(val_loss, 4)}'
        self.update()