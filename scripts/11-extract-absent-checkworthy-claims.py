import argparse
from pathlib import Path

import pandas as pd

from utils import full_dataset_dir, iterate_full_event_files, reduced_dataset_dir, filter_tsv, iterate_reduced_event_files, \
  filename_without_extension


def investigate_absent_checkworthy_claims():
  args = process_command_line_args()
  pd.set_option('display.max_colwidth', None)

  for subset in ['train', 'dev', 'test']:
    print(f'Processing {subset}')
    absent = collect_absent(args.data_dir, subset)
    print(absent)
    print('\n\n\n\n\n')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def collect_absent(data_dir, subset):
  checkworthy_full = collect_checkworthy_claims_from_full_dataset(data_dir, subset)
  print(f'Checkworthy in full - {len(checkworthy_full)}')
  checkworthy_reduced = collect_checkworthy_claims_from_reduced_dataset(data_dir, subset)
  print(f'Checkworthy in reduced - {len(checkworthy_reduced)}')
  absent = find_absent(checkworthy_full, checkworthy_reduced)
  print(f'Absent in reduced - {len(absent)}')
  return absent

def collect_checkworthy_claims_from_full_dataset(data_dir, subset):
  all_dataframes = []
  def collect_checkworthy_claims_for_event(subset_dir, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns=['line_number', 'speaker', 'sentence', 'is_claim']
    df['event_name'] = df.apply(lambda row: filename_without_extension(file_name), axis=1)
    df = df[df['is_claim'] == 1]
    all_dataframes.append(df)

  events_dir = full_dataset_dir(data_dir, subset)
  iterate_full_event_files(events_dir, collect_checkworthy_claims_for_event)

  return pd.concat(all_dataframes).reset_index(drop=True)

def collect_checkworthy_claims_from_reduced_dataset(data_dir, subset):
  all_dataframes = []
  def collect_checkworthy_claims_for_audio_file_transcript(event_name, event_dir, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['event_name'] = df.apply(lambda row: event_name, axis=1)
    df = df[df['is_claim'] == 1]
    all_dataframes.append(df)

  iterate_reduced_event_files(reduced_dataset_dir(data_dir) / '07-sentence-level-alignment' / subset,
    collect_checkworthy_claims_for_audio_file_transcript,
    fn_file_filter=filter_tsv
  )

  # omitting the line_number on purpose - single event could be separated into several files
  # due to event recording being split into several parts
  return pd.concat(all_dataframes).reset_index(drop=True)[['speaker', 'sentence', 'is_claim', 'event_name']]

def find_absent(checkworthy_full, checkworthy_reduced):
  def is_absent(row):
    reduced_filtered = checkworthy_reduced.loc[
      (checkworthy_reduced['speaker'] == row['speaker']) &
      (checkworthy_reduced['sentence'] == row['sentence'].strip()) &
      (checkworthy_reduced['event_name'] == row['event_name'])
    ]
    return len(reduced_filtered) == 0

  return checkworthy_full[checkworthy_full.apply(is_absent, axis=1)]

if __name__ == '__main__':
  investigate_absent_checkworthy_claims()
