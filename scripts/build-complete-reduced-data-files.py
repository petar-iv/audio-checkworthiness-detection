import argparse
from pathlib import Path

import pandas as pd

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, filename_without_extension, sentence_audio_base_name


def build_complete_reduced_data_files():
  args = process_command_line_args()

  dataset_dir = reduced_dataset_dir(args.data_dir)

  for subset_pair in [
      ('train', args.train_dir_name),
      ('dev', args.dev_dir_name),
      ('test', args.test_dir_name)
    ]:
    create_files_for_subset(dataset_dir, subset_pair[0], subset_pair[1])

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--train_dir_name', type=str, default='train')
  parser.add_argument('--dev_dir_name', type=str, default='dev')
  parser.add_argument('--test_dir_name', type=str, default='test')
  return parser.parse_args()

def create_files_for_subset(dataset_dir, source_subset, target_subset):

  def build_file_path(dir_name, event_name, audio_base_name, line_number, extension):
    return str(dataset_dir / dir_name / source_subset / event_name / f'{sentence_audio_base_name(audio_base_name, line_number)}') + extension

  def build_numpy_file_path(dir_name, event_name, audio_base_name, line_number):
    return build_file_path(dir_name, event_name, audio_base_name, line_number, '.npy')

  def process_event_audio_sentence_alignment(event_name, event_dir, file_name, file_path):
    audio_base_name = filename_without_extension(file_name)

    df = pd.read_csv(file_path, sep='\t')
    df = df[['line_number', 'speaker', 'sentence', 'is_claim']]

    df['audio'] = df.apply(lambda row: build_file_path('08-audio-segments', event_name, audio_base_name, row['line_number'], '.wav'), axis=1)
    df['audio-rn'] = df.apply(lambda row: build_file_path('09-audio-segments-rn', event_name, audio_base_name, row['line_number'], '.wav'), axis=1)
    df['audio-gs'] = df.apply(lambda row: build_file_path('10-audio-segments-gs', event_name, audio_base_name, row['line_number'], '.wav'), axis=1)

    column_to_dir = {
      'opensmile-mfcc': 'features/opensmile-mfcc',
      'compare-2013': 'features/compare-2013',
      'compare-2016': 'features/compare-2016',
      'ivector': 'features/ivectors',
      'openl3': 'features/openl3',

      'opensmile-mfcc-rn': 'features-rn/opensmile-mfcc',
      'compare-2013-rn': 'features-rn/compare-2013',
      'compare-2016-rn': 'features-rn/compare-2016',
      'ivector-rn': 'features-rn/ivectors',
      'openl3-rn': 'features-rn/openl3',

      'opensmile-mfcc-gs': 'features-gs/opensmile-mfcc',
      'compare-2013-gs': 'features-gs/compare-2013',
      'compare-2016-gs': 'features-gs/compare-2016',
      'ivector-gs': 'features-gs/ivectors',
      'openl3-gs': 'features-gs/openl3',

      'compare-2016-best-256': 'features/compare-2016-best-256',
      'ner': 'features/ner',

      'bert': 'features/bert/vector',
      'bert-classification': 'features/bert/classification',

      'hubert': 'features/hubert/vector',
      'hubert-classification': 'features/hubert/classification',

      'data2vec-aligned': 'features/data2vec-aligned/vector',
      'data2vec-classification-aligned': 'features/data2vec-aligned/classification'
    }

    for column_name, dir_name in column_to_dir.items():
      df[column_name] = df.apply(lambda row: build_numpy_file_path(dir_name, event_name, audio_base_name, row['line_number']), axis=1)

    event_data_files_dir = dataset_dir / 'data-files' / target_subset / event_name
    event_data_files_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(event_data_files_dir / f'{filename_without_extension(file_name)}.tsv', sep='\t', index=False)

  iterate_reduced_event_files(dataset_dir / '07-sentence-level-alignment' / target_subset,
    process_event_audio_sentence_alignment,
    fn_file_filter=filter_tsv)

if __name__ == '__main__':
  measure_time('building complete reduced data files', build_complete_reduced_data_files)
