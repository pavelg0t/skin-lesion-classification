import torch

from .factory import ModelFactory


@ModelFactory.register('custom2')
class Custom2(torch.nn.Module):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super.__init__(args, kwargs)
