import argparse
from pathlib import Path

import pandas as pd

from utils import reduced_dataset_dir, iterate_reduced_event_files, filter_tsv


def duplicate_train_checkworthy():
  args = process_command_line_args()
  sentence_level_alignment = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment'

  original_train_dir = sentence_level_alignment / 'train'
  print_stats(original_train_dir)

  target_train_dir = sentence_level_alignment / f'train-{args.num_duplicates}x'
  duplicate_checkworthy(original_train_dir, target_train_dir, args.num_duplicates, args.shuffle_seed)
  print_stats(target_train_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--num_duplicates', required=True, type=int)
  parser.add_argument('--shuffle_seed', type=int, default=42)
  return parser.parse_args()

def duplicate_checkworthy(source_dir, target_dir, num_duplicates, seed):
  def count_sentences_in_file(event_name, event_path, file_name, file_path):
    df = pd.read_csv(file_path, sep='\t')
    checkworthy = df[df['is_claim'] == 1]

    for i in range(num_duplicates):
      df = pd.concat([df, checkworthy])

    df = df.sample(frac=1, random_state=seed)

    event_dir = target_dir / event_name
    event_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(str(event_dir / file_name), sep='\t', index=False)

  iterate_reduced_event_files(source_dir, count_sentences_in_file, fn_file_filter=filter_tsv)

def print_stats(train_dir_path):
  checkworthy, non_checkworthy = count_original_sentences(train_dir_path)
  print(f'Checkworthy: {checkworthy}')
  print(f'Non-checkworthy: {non_checkworthy}')
  print(f'Non-checkworthy are {"{:.2f}".format(non_checkworthy/checkworthy)} more than the checkworthy sentences')

def count_original_sentences(train_dir):
  check_worthy_total_count, non_check_worthy_total_count = 0, 0

  def count_sentences_in_file(event_name, event_path, file_name, file_path):
    nonlocal check_worthy_total_count, non_check_worthy_total_count

    df = pd.read_csv(file_path, sep='\t')
    check_worthy_total_count += len(df[df['is_claim'] == 1])
    non_check_worthy_total_count += len(df[df['is_claim'] == 0])

  iterate_reduced_event_files(train_dir, count_sentences_in_file, fn_file_filter=filter_tsv)

  return check_worthy_total_count, non_check_worthy_total_count

if __name__ == '__main__':
  duplicate_train_checkworthy()
