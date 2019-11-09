import torch
from collections import defaultdict

class SessionSaver:
    """ A class for saving pytorch training sesssion
    It requires a full path (path + filename) where to save the training sesssion file.
    """
    def __init__(self, path, frequency = 1, bestonly = True, overwrite = True, metric = 'loss', ascending = False, beta = .9, patience = 0):
        if path.endswith('.tar'):
            self.path = path[:-4]
        else:
            self.path = path
        self.frequency = frequency
        self.bestonly = bestonly
        self.overwrite = True if bestonly else overwrite
        self.metric = metric
        self.direction = -1 if ascending else 1
        self.bestmetric = defaultdict(float)
        self.lastmetric = defaultdict(float)
        self.beta = beta
        self.patience = patience if bestonly else 0


    def save(self, session, epoch, metrics):
        if epoch % self.frequency != 0:
            return
        newmetric = defaultdict(float)
        newmetric['epoch'] = epoch
        for key, val in metrics.items():
            newmetric[key] = self.lastmetric[key] + (val - self.lastmetric[key]) * self.beta
        self.lastmetric = newmetric
        if not self.bestonly:
            if self.overwrite:
                fname = self.path+'.tar'
            else:
                fname = f'{self.path}_{epoch}.tar'
            self.bestmetric = newmetric.copy()
            torch.save(session.state_dict(), fname)
        elif self.metric not in self.bestmetric or newmetric[self.metric]*self.direction < self.bestmetric[self.metric]*self.direction:
            self.bestmetric = newmetric.copy()
            fname = self.path+'.tar'
            torch.save(session.state_dict(), fname)
        if self.patience > 0 and epoch - self.bestmetric['epoch'] > self.patience:
            return False
        else:
            return True

    def state_dict(self):
        state = {'bestmetric':self.bestmetric,
            'lastmetric': self.lastmetric,
            'path': self.path,
            'frequency':self.frequency,
            'overwrite':self.overwrite,
            'bestonly':self.bestonly,
            'beta':self.beta,
            'metric': self.metric,
            'patience': self.patience,
            'direction': self.direction}
        return state

    def load_state_dict(self, state):
        self.bestmetric = state['bestmetric']
        self.lastmetric = state['lastmetric']
        self.path = state['path']
        self.frequency = state['frequency']
        self.overwrite = state['overwrite']
        self.bestonly = state['bestonly']
        self.metric = state['metric']
        self.direction = state['direction']
        self.beta = state['beta']
        self.patience = state['patience']
