import argparse
from pathlib import Path

import pandas as pd

from utils import reduced_dataset_dir, iterate_reduced_event_files, filter_tsv


def balance_dataset():
  args = process_command_line_args()
  balance_train(args)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--non_checkworthy_to_checkworthy', type=int, default=1)
  parser.add_argument('--seed', type=int, default=42)
  return parser.parse_args()

def balance_train(args):
  source_dir = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / 'train'
  target_dir = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / f'train-balanced-{args.non_checkworthy_to_checkworthy}'

  def process_event_audio_transcript(event_name, event_path, file_name, file_path):
    target_event_dir = target_dir / event_name
    target_event_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path, sep='\t')
    check_worthy = df[df['is_claim']==1]
    non_check_worthy = df[df['is_claim']==0]
    non_check_worthy = non_check_worthy.sample(n=args.non_checkworthy_to_checkworthy * len(check_worthy), random_state=args.seed)

    balanced = pd.concat([check_worthy, non_check_worthy])
    if len(balanced) > 0:
      balanced.to_csv(target_event_dir / file_name, sep='\t')

  iterate_reduced_event_files(source_dir, process_event_audio_transcript, filter_tsv)

if __name__ == '__main__':
  balance_dataset()
