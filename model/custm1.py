import torch

from .factory import ModelFactory


@ModelFactory.register('custom1')
class Custom1(torch.nn.Module):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super.__init__(args, kwargs)
