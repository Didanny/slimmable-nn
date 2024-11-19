import torch
import contextlib
import time

from typing import Optional

class Profile(contextlib.ContextDecorator):
    def __init__(self, t: Optional[float] = 0.0):
        super().__init__()
        
        self.t = t 
        self.cuda = torch.cuda.is_available()
        
    def __enter__(self):
        self.start = self.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt
        
    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()