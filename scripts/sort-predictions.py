import re
import argparse
from pathlib import Path
from os.path import basename

import pandas as pd

from utils import reduced_dataset_dir


def sort_predictions():
  args = process_command_line_args()
  df = load_senetences(args)
  print_predictions(df, args.predictions_file)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=Path, required=True)
  parser.add_argument('--predictions_file', type=Path, required=True)
  parser.add_argument('--subset', required=True, type=str, choices=['dev', 'test'])
  return parser.parse_args()

def load_senetences(args):
  file_identifier = basename(args.predictions_file)
  event_name, file_name = split_file_identifier(file_identifier)
  return pd.read_csv(reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / args.subset / event_name / f'{file_name}.tsv', sep='\t')

def split_file_identifier(file_identifier):
  return re.search('(.+)-(audio-\d+)', file_identifier).groups()

def print_predictions(df, file_path):
  predictions_map = {}
  with open(file_path, 'r') as file:
    for line in file:
      [line_number, probability] = line.split()
      line_number = int(line_number)
      probability = float(probability)
      predictions_map[line_number] = probability

  sorted_predictions = sorted(predictions_map.items(), key=lambda x: x[1], reverse=True)
  counter = 1
  for line_number, prediction in sorted_predictions:
    row = df[df['line_number'] == line_number]
    speaker = row['speaker'].item()
    sentence = row['sentence'].item()
    print(f'{str(counter).zfill(3)} | {line_number} {prediction} {speaker} {sentence}')
    counter += 1

if __name__ == '__main__':
  sort_predictions()
