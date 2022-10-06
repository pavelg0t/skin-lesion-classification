import logging
from typing import Callable

import timm
from torch.nn import Module

logger = logging.getLogger(__name__)


class ModelFactory:
    """ The factory class for creating nn models"""

    registry = {}
    """ Internal registry for available nn models """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Module class to the internal registry.
        Args:
            name (str): The name of the model.
        Returns:
            The Module class itself.
        """

        def inner_wrapper(wrapped_class: Module) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Model %s already exists. Will replace it', name,
                )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_model(cls, name: str, **kwargs) -> 'Module':
        """ Factory command to create the nn model.
        This method gets the appropriate nn model class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the nn model to create.
        Returns:
            An instance of the nn model that is created.
        """

        if name in cls.registry:
            model_class = cls.registry[name]
            model = model_class(name, **kwargs)
            return model
        elif name in timm.list_models():
            pretrained = kwargs.get('pretrained', True)
            n_class = kwargs.get('n_class')
            return timm.create_model(name, pretrained=pretrained, num_classes=n_class)
        else:
            logger.warning(
                'Model %s does not exist in the registry nor timm library', name,
            )
            return None

    @classmethod
    def is_model(cls, name: str) -> bool:
        """ Factory command to check if a particular nn model exists.
        This method checks if a particular nn model class exists in the registry.
        Args:
            name (str): The name of the nn model to create.
        Returns:
            Boolean value.
        """

        return name in cls.registry
