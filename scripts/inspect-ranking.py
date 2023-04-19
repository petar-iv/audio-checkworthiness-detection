import argparse
from pathlib import Path

import pandas as pd

from utils import reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, filename_without_extension


def inspect_ranking():
  args = process_command_line_args()

  actual_dir = reduced_dataset_dir(args.data_dir) / '07-sentence-level-alignment' / args.actual_dir_name

  def iterate_actual(event_name, event_path, file_name, file_path):
    print(f'\n---------- Printing stats on {event_name} ----------')
    df_actual = pd.read_csv(file_path, sep='\t')
    df_predicted = pd.read_csv(args.predictions_dir / f'{event_name}-{filename_without_extension(file_name)}.txt', sep=' ', names=['line_number', 'score'])
    assert len(df_actual) == len(df_predicted)
    result = df_actual.join(df_predicted.set_index('line_number'), on='line_number', how='inner', lsuffix='_left')
    assert len(result) == len(df_actual)

    result = result.sort_values(by=['score'], ascending=False)
    result['rank'] = range(1, len(result) + 1)
    result = result[['rank', 'line_number', 'speaker', 'sentence', 'score', 'is_claim']]

    print(f'Number of sentences: {len(result)}')
    checkworthy = result[result['is_claim'] == 1]
    print(f'Number of checkworthy sentences: {len(checkworthy)}')
    print(f'Ranks of checkworthy: {checkworthy["rank"].tolist()}')

    print('\nCheckworthy sentences and their ranks:')
    for _, row in checkworthy.iterrows():
      print(f'Rank #{row["rank"]}: {row["speaker"]}, {row["sentence"]}')

    print(f'\nTop {args.sentences_to_display} ranked:')
    for _, row in result[0:args.sentences_to_display].iterrows():
      print(f'Rank #{row["rank"]}: {row["speaker"]}, {row["sentence"]}, Actually checkworthy? {row["is_claim"] == 1}')

    print(f'\nBottom {args.sentences_to_display} ranked:')
    for _, row in result[-args.sentences_to_display:].iterrows():
      print(f'Rank #{row["rank"]}: {row["speaker"]}, {row["sentence"]}')


    print('\n\n\n')

  iterate_reduced_event_files(actual_dir, iterate_actual, filter_tsv)

def process_command_line_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=Path, required=True)
  parser.add_argument('--actual_dir_name', type=Path, required=True)
  parser.add_argument('--predictions_dir', type=Path, required=True)
  parser.add_argument('--sentences_to_display', type=int, default=5)

  return parser.parse_args()

if __name__ == '__main__':
  inspect_ranking()
