import argparse
from pathlib import Path

import pandas as pd

from utils import full_dataset_dir, iterate_full_event_files, filename_without_extension


def extract_system_sentences():
  args = process_command_line_args()

  checkworthy = []
  non_checkworthy = []

  target_dir = full_dataset_dir(args.data_dir) / 'system-sentences'
  target_dir.mkdir(parents=True, exist_ok=True)

  for subset in ['test', 'dev', 'train']:
    print(f'Processing {subset}')
    checkworthy_in_subset, non_checkworthy_in_subset = extract_system_sentences_in_subset(args.data_dir, subset)
    checkworthy.append(checkworthy_in_subset)
    non_checkworthy.append(non_checkworthy_in_subset)

  pd.concat(checkworthy).reset_index(drop=True).to_csv(target_dir / 'checkworthy.tsv', sep='\t')
  pd.concat(non_checkworthy).reset_index(drop=True).to_csv(target_dir / 'non_checkworthy.tsv', sep='\t')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def extract_system_sentences_in_subset(data_dir, subset):
  checkworthy = []
  non_checkworthy = []

  def extract_system_sentences_for_event(subset_dir, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns=['line_number', 'speaker', 'sentence', 'is_claim']
    df['subset'] = subset
    df['event_name'] = df.apply(lambda row: filename_without_extension(file_name), axis=1)
    df = df[['subset', 'event_name', 'line_number', 'speaker', 'sentence', 'is_claim']]

    df = df[df['speaker'] == 'SYSTEM']

    checkworthy.append(df[df['is_claim'] == 1])
    non_checkworthy.append(df[df['is_claim'] == 0])

  events_dir = full_dataset_dir(data_dir, subset)
  iterate_full_event_files(events_dir, extract_system_sentences_for_event)

  checkworthy = pd.concat(checkworthy).reset_index(drop=True)
  non_checkworthy = pd.concat(non_checkworthy).reset_index(drop=True)

  return checkworthy, non_checkworthy

if __name__ == '__main__':
  extract_system_sentences()
