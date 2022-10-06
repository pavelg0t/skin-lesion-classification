import argparse
from os.path import exists

import timm

from dataset.factory import DatasetFactory
from dataset.factory import TransformFactory
from model.factory import ModelFactory


class ValidateModelName(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        if (values not in timm.list_models()) and (not ModelFactory.is_model(values)):
            parser.error(f'Please enter a valid model name. Got: {values}')
        setattr(namespace, self.dest, values)


class ValidateDatasetName(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        if not DatasetFactory.is_dataset(values):
            parser.error(f'Please enter a valid dataset name. Got: {values}')
        setattr(namespace, self.dest, values)


class ValidateTransformType(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        if not TransformFactory.is_transform(values):
            parser.error(f'Please enter a valid transform type. Got: {values}')
        setattr(namespace, self.dest, values)


class CheckIfFileExists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        if not exists(values):
            parser.error(f'Please enter an existing file path. Got: {values}')
        setattr(namespace, self.dest, values)
