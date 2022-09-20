import argparse
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

from utils.datasets import ISIC2020
from utils.datasets import TrainDataset


def get_timm_model(m_name, pretrained, num_classes):

    model = timm.create_model(m_name, pretrained, num_classes)
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)
    inpt_dim = data_config['input_size'][-1]

    return model, transform, inpt_dim


def train_isic2020():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # parse ISIC2020 filenames
    isic2020_data = ISIC2020(
        base_dir=train_args.train_path, labels_path=train_args.labels_path,
    )

    # get timm model
    bb_model, transform, inpt_dim = get_timm_model(
        m_name=train_args.m_name,
        pretrained=True,
        num_classes=train_args.n_class,
    )

    # Preprocessing that will be run on each individuall train image

    def ppc_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return transform(img)

    # dataloader parameters
    params = {
        'batch_size': train_args.batch_size,
        'shuffle': True,
        'num_workers': train_args.num_workers,
    }

    data_loader = torch.utils.data.DataLoader(
        TrainDataset(
            file_paths=isic2020_data.image_files,
            labels=isic2020_data.image_categories,
            preproc=ppc_image,
        ),
        **params,
    )

    bb_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bb_model.parameters(), lr=0.001, momentum=0.9)

    # some addtional info
    time_start = time.time()

    # with torch.set_grad_enabled(True):
    for epoch in range(train_args.n_epoch):

        running_loss = 0.0
        n_mini_batch = math.ceil(
            len(isic2020_data.image_files) / train_args.batch_size,
        )

        for i, local_batch in tqdm(enumerate(data_loader), leave=True, total=n_mini_batch):

            image_batch = local_batch['image'].to(device)
            labels = local_batch['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = bb_model(image_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print_freq = math.ceil(n_mini_batch/10)
            # print every 10% of the mini-batches
            if i % print_freq == (print_freq-1):
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f}',
                )
                running_loss = 0.0

    time_needed = np.round((time.time() - time_start), 2)
    m_params = sum(
        torch.tensor(param.shape).prod()
        for param in bb_model.parameters()
    )

    # print metadata
    print(f'test time: {time_needed}')
    print(f'input size: {inpt_dim}')
    print(f'model params: {m_params}')


def main():

    train_isic2020()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        default=None,  # required
        help='Train data path',
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        default=None,  # required
        help='Labels JOSN file path',
    )
    parser.add_argument(
        '--m_name',
        type=str,
        default=None,  # required
        help='Model name',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers used for data preprocessing',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size',
    )
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=20,
        help='Number of epochs to run',
    )
    parser.add_argument(
        '--n_class',
        type=int,
        default=2,
        help='Number of data classes',
    )

    train_args, unparsed = parser.parse_known_args()
    main()
