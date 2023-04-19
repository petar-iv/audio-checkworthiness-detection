import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import measure_time, reduced_dataset_dir, build_dataframe_for_reduced_dataset


def train_scalers():
  np.set_printoptions(threshold=10)
  args = process_command_line_args()

  df = build_dataframe_for_reduced_dataset(reduced_dataset_dir(args.data_dir) / 'data-files' / 'train')

  for df_column in args.df_columns:
    def train_scaler_for_column():
      train_samples = collect_samples(df[df_column].tolist())
      scaler = train_scaler_with_samples(train_samples)
      save_scaler(args.target_dir, df_column, scaler)
      print_info(scaler, train_samples[0])

    measure_time(f'training scaler for data in column {df_column}', train_scaler_for_column)
    print('\n\n\n\n\n')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--target_dir', required=True, type=Path)
  parser.add_argument('--df_columns', required=True, type=str, nargs='+')
  return parser.parse_args()

def train_scaler_with_samples(train_samples):
  scaler = MinMaxScaler()
  return scaler.fit(train_samples)

def collect_samples(vector_files):
  samples = []
  for vector_file in vector_files:
    v = np.load(vector_file)
    if len(v.shape) == 1:
      samples.append(v.tolist())
    elif len(v.shape) == 2:
      for sequence_element in v:
        samples.append(sequence_element.tolist())
    else:
      raise Exception(f'No support for shape {v.shape}')
  return samples

def save_scaler(target_dir, feature_set, scaler):
  target_dir.mkdir(parents=True, exist_ok=True)
  torch.save(scaler, target_dir / f'{feature_set}.bin')

def print_info(scaler, sample):
  print(f'data_min_: {scaler.data_min_}')
  print(f'data_max_: {scaler.data_max_}\n')
  sample = np.array(sample)
  print(f'Sample: {sample}')
  print(f'Scaled sample: {scaler.transform([sample])}')

if __name__ == '__main__':
  measure_time('training scalers', train_scalers)
