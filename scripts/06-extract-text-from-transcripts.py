import argparse
from pathlib import Path

import pandas as pd

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, filename_without_extension


def extract_text_from_transcripts():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    extract_text_from_transcripts_for_all_events(args, subset)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def extract_text_from_transcripts_for_all_events(args, subset):
  events_dir = reduced_dataset_dir(args.data_dir) / '04-audio-level-pairs' / subset
  iterate_reduced_event_files(events_dir, extract_text_from_transcripts_for_event, fn_file_filter=filter_tsv)

def extract_text_from_transcripts_for_event(event_name, event_path, file_name, file_path):
  df = pd.read_csv(file_path, sep='\t')
  sentences = df['sentence'].tolist()

  txt_file = f'{filename_without_extension(file_name)}.txt'
  with open(event_path / txt_file, 'w') as f:
    f.write('\n'.join(sentences))

if __name__ == '__main__':
  measure_time(f'extracting text from transcripts', extract_text_from_transcripts)
