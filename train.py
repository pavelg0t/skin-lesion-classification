import argparse
import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from tqdm import tqdm

import wandb
from dataset import DatasetFactory
from dataset import TransformFactory
from model import ModelFactory
from utils.validate import ValidateDatasetName
from utils.validate import ValidateModelName
from utils.validate import ValidateTransformType

SEED = 42
TRAIN_DATA_RATIO = 0.9


def run_training(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    transform = TransformFactory.create_transform(
        t_type=config.transform, m_name=config.m_name,
    )

    train_dataset = DatasetFactory.create_dataset(
        config.dataset,
        multiclass=config.multiclass,
        train=True,
        split_ratio=TRAIN_DATA_RATIO,
        seed=SEED,
        preproc=transform,
    )

    val_dataset = DatasetFactory.create_dataset(
        config.dataset,
        multiclass=config.multiclass,
        train=False,
        preproc=transform,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ModelFactory.create_model(
        name=config.m_name, n_class=train_dataset.n_class,
    )
    model.to(device)

    # This should be placed in a separate folder 'Loss_function' ?
    def weights_for_imbalanced_cats(cats):
        u_cats = np.unique(cats)
        weights = np.zeros(shape=u_cats.shape)
        for i, cat in enumerate(u_cats.tolist()):
            weights[i] = np.sum(cats == cat) / cats.shape
        return torch.div(np.amax(weights), torch.Tensor(weights))

    # It should be parametrized!!!!!!!!!!!!!!!!!!!!!
    criterion = nn.CrossEntropyLoss(
        weight=weights_for_imbalanced_cats(train_dataset.image_categories),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # some addtional info
    time_start = time.time()

    for epoch in range(config.n_epoch):

        train_running_loss = 0.0
        val_running_loss = 0.0
        train_mini_batch = math.ceil(
            len(train_dataset.image_files) / config.batch_size,
        )
        val_mini_batch = math.ceil(
            len(val_dataset.image_files) / config.batch_size,
        )

        train_acc = Accuracy(num_classes=train_dataset._n_class, average=None)
        val_acc = Accuracy(num_classes=val_dataset._n_class, average=None)

        # Train steps
        model.train()
        for i, local_batch in tqdm(
            enumerate(train_data_loader), leave=True, total=train_mini_batch,
        ):

            train_image_batch = local_batch['image'].to(device)
            train_category = local_batch['category'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(train_image_batch)
            train_loss = criterion(outputs, train_category)
            train_loss.backward()
            optimizer.step()

            # print statistics
            train_running_loss += train_loss.item()
            _ = train_acc(outputs, train_category)

            print_freq = math.ceil(train_mini_batch/100)
            # print every 1% of the mini-batches
            if i % print_freq == (print_freq-1):
                train_acc_metrics = {
                    f'train/accuracy_{k}': v for k, v
                    in enumerate(train_acc.compute().tolist())
                }
                train_metrics = {
                    'train/epoch':
                        (i + 1 + (epoch*train_mini_batch)) / train_mini_batch,
                    'train/running_loss': train_running_loss / print_freq,
                    'train/example_ct': print_freq,
                }

                train_metrics.update(train_acc_metrics)

                wandb.log(train_metrics)
                train_running_loss = 0.0
                train_acc.reset()

        train_acc.reset()

        # Validation step
        model.eval()
        for i, local_batch in tqdm(
            enumerate(val_data_loader), leave=True, total=val_mini_batch,
        ):
            val_image_batch = local_batch['image'].to(device)
            val_category = local_batch['category'].to(device)

            # forward + backward + optimize
            outputs = model(val_image_batch)
            val_loss = criterion(outputs, val_category)

            # print statistics
            val_running_loss += val_loss.item()
            _ = val_acc(outputs, val_category)

        val_acc_metrics = {
            f'val/accuracy_{k}': v for k, v
            in enumerate(val_acc.compute().tolist())
        }
        val_metrics = {
            'val/epoch': epoch,
            'val/val_running_loss':
                val_running_loss / len(val_dataset.image_files),
        }
        val_metrics.update(val_acc_metrics)
        wandb.log(val_metrics)
        val_acc.reset()

    time_needed = np.round((time.time() - time_start), 2)
    m_params = sum(
        torch.tensor(param.shape).prod()
        for param in model.parameters()
    )

    # print metadata
    print(f'test time: {time_needed}')
    # print(f'input size: {inpt_dim}')
    print(f'model params: {m_params}')


def main(args):

    # login to wandb
    wandb.login()

    # initialise a wandb run
    wandb.init(
        project=args.prj_name,
        name=f'run-{datetime.now()}'.replace(' ', '-'),
        config={k: v for k, v in vars(args).items() if k != 'prj_name'},
    )

    # Copy your config
    config = wandb.config

    run_training(config)

    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'prj_name', type=str,
        help='Weights and Biases project name',
    )
    parser.add_argument(
        '--m_name', action=ValidateModelName, required=True,
        help='Model name',
    )
    parser.add_argument(
        '--dataset', action=ValidateDatasetName, required=True,
        help='Dataset Name',
    )
    parser.add_argument(
        '--multiclass', action='store_true', default=False,
        help='Treat data as a multi-class or binary',
    )
    parser.add_argument(
        '--transform', action=ValidateTransformType, required=True,
        help='Data pre-processing type',
    )
    # TO DO: Select the LossFunction (CostFunction)
    # -> Do Validate if it exists in the moudels

    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of workers used for data preprocessing',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size',
    )
    parser.add_argument(
        '--n_epoch', type=int, default=20,
        help='Number of epochs to run',
    )

    args = parser.parse_args()
    main(args)
