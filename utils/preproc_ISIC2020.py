import argparse
import json
import pandas as pd
import os

def create_category_map():

    df = pd.read_csv(args.train_gt)
    df = df[['image_name', 'target']]
    df = df.to_json(orient="values")
    df = json.loads(df)
    df = dict(df)

    src_base_dir = '/'.join(args.train_gt.split('/')[:-1])
    dest_filename = os.path.join(src_base_dir, 'ISIC2020_train_map.json')

    with open(dest_filename, 'w') as f:
        json.dump(df, f, indent=4)

def create_duplicates_json():
    df = pd.read_csv(args.train_dupl)
    df = df.to_json(orient="values")
    df = json.loads(df)
    df = dict(df)

    src_base_dir = '/'.join(args.train_gt.split('/')[:-1])
    dest_filename = os.path.join(src_base_dir, 'ISIC2020_train_dupl.json')

    with open(dest_filename, 'w') as f:
        json.dump(df, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train_gt',
      type=str,
      default=None, #required
      help='Training data ground truth CSV file path'
    )
    parser.add_argument(
      '--train_dupl',
      type=str,
      default=None, #required
      help='Training data duplicates CSV file path'
    )

    args, unparsed = parser.parse_known_args()

    create_category_map()
    create_duplicates_json()

