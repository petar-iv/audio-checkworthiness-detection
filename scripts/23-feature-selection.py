import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, \
  sentence_audio_base_name, filename_without_extension

def select_features():
  args = process_command_line_args()

  train_identifiers, train_features, train_is_claim = load_data_for_subset(args, 'train')
  transforming = select_best(args, train_features, train_is_claim)
  train_features = transforming.transform(train_features)
  save_features(args, 'train', train_identifiers, train_features)

  dev_identifiers, dev_features, _ = load_data_for_subset(args, 'dev')
  dev_features = transforming.transform(dev_features)
  save_features(args, 'dev', dev_identifiers, dev_features)

  test_identifiers, test_features, _ = load_data_for_subset(args, 'test')
  test_features = transforming.transform(test_features)
  save_features(args, 'test', test_identifiers, test_features)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--features', required=True, type=str)
  parser.add_argument('--scaler', required=True, type=str)
  parser.add_argument('--num_features', required=True, type=int)
  parser.add_argument('--target_dir', required=True, type=Path)
  return parser.parse_args()

def load_data_for_subset(args, subset):
  identifiers = []
  features = []
  is_claim = []

  scaler = torch.load(args.scaler)

  def add_samples(event_name, event_dir, file_name, file_path):
    audio_base_name = filename_without_extension(file_name)
    df = pd.read_csv(file_path, sep='\t')
    for _, row in df.iterrows():
      line_number = row['line_number']

      identifiers.append({
        'event': event_name,
        'audio_base_name': audio_base_name,
        'line_number': line_number
      })

      features_file = reduced_dataset_dir(args.data_dir) / args.features / subset / event_name / f'{sentence_audio_base_name(audio_base_name, line_number)}.npy'
      v = np.load(features_file)
      if len(v.shape) != 1:
        raise Exception('Only 1D vectors allowed for feature selection')
      v = scaler.transform([v])[0]
      features.append(v.tolist())

      is_claim.append(row['is_claim'])

  iterate_reduced_event_files(reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / subset,
    add_samples, fn_file_filter=filter_tsv)

  return identifiers, np.array(features), np.array(is_claim)

def select_best(args, features, is_claim):
  return SelectKBest(score_func=chi2, k=args.num_features).fit(features, is_claim)

def save_features(args, subset, identifiers, features):
  for (sample_identifiers, sample_features) in zip(identifiers, features):
    dir_path = reduced_dataset_dir(args.data_dir) / args.target_dir / subset / sample_identifiers['event']
    dir_path.mkdir(exist_ok=True, parents=True)
    np.save(dir_path / f'{sentence_audio_base_name(sample_identifiers["audio_base_name"], sample_identifiers["line_number"])}.npy', sample_features)

if __name__ == '__main__':
  measure_time('selecting best features', select_features)
