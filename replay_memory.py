from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, next_state, reward):
        """Saves a transition."""
        self.memory.append(Transition(state, action, mask, next_state, reward))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
