class PhaseEarlyStopping:
    def __init__(self, patience=20, active_phase='phase3', enabled=True):
        self.patience = int(patience)
        self.active_phase = active_phase
        self.enabled = enabled
        self.best = None
        self.bad_epochs = 0

    def update(self, epoch, phase, score):
        if not self.enabled or phase != self.active_phase:
            return False
        if self.best is None or score > self.best:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
