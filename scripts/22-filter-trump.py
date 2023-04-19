import argparse
from pathlib import Path

import pandas as pd

from utils import reduced_dataset_dir, iterate_reduced_event_files, filter_tsv


def filter_trump():
  args = process_command_line_args()
  sentence_level_alignment = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment'

  for subset in ['train', 'dev', 'test']:
    filter_trump_in_subset(sentence_level_alignment, subset)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def filter_trump_in_subset(sentence_level_alignment, subset):
  def filter_trump_in_file(event_name, event_path, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df.apply(speaker_is_donald_trump, axis=1)]

    if len(df) > 0:
      event_dir = sentence_level_alignment / f'trump-{subset}' / event_name
      event_dir.mkdir(parents=True, exist_ok=True)
      df.to_csv(str(event_dir / file_name), sep='\t', index=False)

  iterate_reduced_event_files(sentence_level_alignment / subset, filter_trump_in_file, fn_file_filter=filter_tsv)

def speaker_is_donald_trump(row):
  speaker = row['speaker']
  speaker = speaker.lower()
  return 'trump' in speaker and 'melania' not in speaker

if __name__ == '__main__':
  filter_trump()
