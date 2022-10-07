import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Callable

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetFactory:
    """ The factory class for creating datasets"""

    # Internal registry for available datasets
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Dataset class to the internal registry.
        Args:
            name (str): The name of the dataset.
        Returns:
            The Dataset class itself.
        """

        def inner_wrapper(wrapped_class: Dataset) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Dataset %s already exists. Will replace it', name,
                )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> Dataset:
        """ Factory command to create the dataset.
        This method gets the appropriate Dataset class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the dataset to create.
        Returns:
            An instance of the Dataset that is created.
        """

        if name not in cls.registry:
            logger.warning('Dataset %s does not exist in the registry', name)
            return None

        dataset_class = cls.registry[name]
        dataset = dataset_class(**kwargs)
        return dataset

    @classmethod
    def is_dataset(cls, name: str) -> bool:
        """ Factory command to check if a particular dataset exists.
        This method checks if a particular dataset class exists in the registry.
        Args:
            name (str): The name of the dataset to create.
        Returns:
            Boolean value.
        """

        return name in cls.registry


class TransformFactory:
    """ The factory class for creating image pre-processing steps"""

    # Internal registry for available image transform steps
    registry = {}

    @classmethod
    def register(cls, t_type: str) -> Callable:
        """ Class method to register transform steps class to the internal registry.
        Args:
            t_type (str): The type of the transform steps.
        Returns:
            The Compose class itself.
        """

        def inner_wrapper(wrapped_class: TransformBase) -> Callable:
            if t_type in cls.registry:
                logger.warning(
                    'Transform %s already exists. Will replace it', t_type,
                )
            cls.registry[t_type] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_transform(cls, t_type: str, **kwargs) -> 'TransformBase':
        """ Factory command to create the Compose instance.
        This method gets the appropriate Compose class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            t_type (str): The type of the transform to create.
        Returns:
            An instance of the Compose that is created.
        """

        if t_type not in cls.registry:
            logger.warning(
                'Transform %s does not exist in the registry', t_type,
            )
            return None

        transform_class = cls.registry[t_type]
        transform = transform_class(**kwargs)
        return transform

    @classmethod
    def is_transform(cls, t_type: str) -> bool:
        """ Factory command to check if a particular transform exists.
        This method checks if a particular transform class exists in the registry.
        Args:
            t_type (str): The type of the transform to create.
        Returns:
            Boolean value.
        """

        return t_type in cls.registry


class TransformBase(metaclass=ABCMeta):
    """ Base class for an image transform object """

    def __init__(self, *args, **kwargs):
        """ Constructor """
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        """ Abstract method to run a command """
        pass
