import argparse
import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb
from dataset import DatasetFactory, TransformFactory
from model import ModelFactory
from utils.validate import ValidateDatasetName
from utils.validate import ValidateModelName
from utils.validate import ValidateTransformType


def run_training(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    transform = TransformFactory.create_transform(
        t_type=config.transform, m_name=config.m_name)

    dataset = DatasetFactory.create_dataset(
        name=config.dataset, preproc=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = ModelFactory.create_model(
        name=config.m_name, n_class=dataset.n_class,
    )
    model.to(device)

    # It should be parametrized!!!!!!!!!!!!!!!!!!!!!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # some addtional info
    time_start = time.time()

    # with torch.set_grad_enabled(True):
    for epoch in range(config.n_epoch):

        running_loss = 0.0
        n_mini_batch = math.ceil(
            len(dataset.image_files) / config.batch_size,
        )

        for i, local_batch in tqdm(enumerate(data_loader), leave=True, total=n_mini_batch):

            image_batch = local_batch['image'].to(device)
            category = local_batch['category']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image_batch)
            loss = criterion(outputs, category)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print_freq = math.ceil(n_mini_batch/100)
            # print every 10% of the mini-batches
            if i % print_freq == (print_freq-1):
                metrics = {
                    'train/epoch': (i + 1 + (epoch * n_mini_batch)) / n_mini_batch,
                    'train/running_loss': running_loss / print_freq,
                    'train/example_ct': print_freq,
                }
                # print(
                #     f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f}',
                )
                wandb.log(metrics)
                running_loss=0.0

    time_needed=np.round((time.time() - time_start), 2)
    m_params=sum(
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
        project = args.prj_name,
        name = f'run-{datetime.now()}'.replace(' ', '-'),
        config = {k: v for k, v in vars(args).items() if k != 'prj_name'},
    )

    # Copy your config
    config=wandb.config

    run_training(config)

    wandb.finish()


if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument(
        'prj_name', type = str,
        help = 'Weights and Biases project name',
    )
    parser.add_argument(
        '--m_name', action = ValidateModelName, required = True,
        help = 'Model name',
    )
    parser.add_argument(
        '--dataset', action = ValidateDatasetName, required = True,
        help = 'Dataset Name',
    )
    parser.add_argument(
        '--transform', action = ValidateTransformType, required = True,
        help = 'Data pre-processing type',
    )
    parser.add_argument(
        '--num_workers', type = int, default = 4,
        help = 'Number of workers used for data preprocessing',
    )
    parser.add_argument(
        '--batch_size', type = int, default = 4,
        help = 'Batch size',
    )
    parser.add_argument(
        '--n_epoch', type = int, default = 20,
        help = 'Number of epochs to run',
    )

    args=parser.parse_args()
    main(args)
