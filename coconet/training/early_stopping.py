import copy


class EarlyStopper:
    """Stops after `patience` epochs without an improvement of more than `min_delta`."""

    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_state = None

    def step(self, value, epoch, model):
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
