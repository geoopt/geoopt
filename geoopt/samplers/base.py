import torch.optim as optim


class Sampler(optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self.n_rejected = 0
        self.steps = 0
        self.burnin = True

        self.log_probs = []
        self.acceptance_probs = []
    
        
    @property
    def rejection_rate(self):
        if self.steps > 0:
            return self.n_rejected / self.steps
        else:
            return 0.0
