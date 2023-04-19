# Code taken from https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/blob/ec6370e19f67f63772aff963cd1ee48284d7a599/task1/baselines/subtask_1b.py

import random
import argparse
from pathlib import Path

import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate import evaluate
from utils import measure_time, set_seed, format_number, \
  full_dataset_dir, build_dataframe_for_full_dataset, \
  reduced_dataset_dir, build_dataframe_for_reduced_dataset


def run_baselines():
  args = process_command_line_args()
  set_seed(args.seed)

  datasets = load_datasets(args)
  measure_time('random baseline', lambda: run_random_baseline(datasets))
  measure_time('ngram baseline', lambda: run_ngram_baseline(datasets))

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--dataset_type', required=True, type=str, choices=['full', 'reduced'])
  parser.add_argument('--train_dir_name', type=str, default='train')
  parser.add_argument('--dev_dir_name', type=str, default='dev')
  parser.add_argument('--test_dir_name', type=str, default='test')
  parser.add_argument('--seed', type=int, default=42)
  return parser.parse_args()

def load_datasets(args):
  if args.dataset_type == 'full':
    return load_full_datasets(args)

  if args.dataset_type == 'reduced':
    return load_reduced_datasets(args)

  raise Exception(f'Cannot handle dataset_type "{args.dataset_type}"')

def load_full_datasets(args):
  return {
    'train': build_dataframe_for_full_dataset(full_dataset_dir(args.data_dir, 'train')),
    'dev': build_dataframe_for_full_dataset(full_dataset_dir(args.data_dir, 'dev')),
    'test': build_dataframe_for_full_dataset(full_dataset_dir(args.data_dir, 'test'))
  }

def load_reduced_datasets(args):
  return {
    'train': build_dataframe_for_reduced_dataset(reduced_dataset_dir(args.data_dir) / 'data-files' / args.train_dir_name),
    'dev': build_dataframe_for_reduced_dataset(reduced_dataset_dir(args.data_dir) / 'data-files' / args.dev_dir_name),
    'test': build_dataframe_for_reduced_dataset(reduced_dataset_dir(args.data_dir) / 'data-files' / args.test_dir_name)
  }

def run_random_baseline(datasets):
  def run_random(sentence):
    return random.random()

  dev_evaluation = run_model_on(run_random, datasets['dev'])
  print_evaluation('dev', dev_evaluation)

  test_evaluation = run_model_on(run_random, datasets['test'])
  print_evaluation('test', test_evaluation)

def run_ngram_baseline(datasets):
  pipeline = Pipeline([
      ('ngrams', TfidfVectorizer(ngram_range=(1, 1))),
      ('clf', SVC(C=1, gamma=0.75, kernel='rbf'))
  ])

  train = datasets['train']
  pipeline.fit(train['sentence'], train['is_claim'])

  def run_ngram(sentence):
    result = pipeline.decision_function(pd.Series([sentence]))
    return result[0]

  dev_evaluation = run_model_on(run_ngram, datasets['dev'])
  print_evaluation('dev', dev_evaluation)

  test_evaluation = run_model_on(run_ngram, datasets['test'])
  print_evaluation('test', test_evaluation)

def run_model_on(fn_run, dataset):
  unique_file_identifiers = dataset['file_identifier'].unique().tolist()

  unique_file_identifiers.sort()
  actual = { file_identifier: {} for file_identifier in unique_file_identifiers }
  predicted = { file_identifier: {} for file_identifier in unique_file_identifiers }

  for _, row in dataset.iterrows():
    sample_file_identifier = row['file_identifier']
    sample_line_number = row['line_number']
    sample_sentence = row['sentence']
    sample_actual = row['is_claim']
    sample_predicted = fn_run(sample_sentence)

    actual[sample_file_identifier][sample_line_number] = sample_actual
    predicted[sample_file_identifier][sample_line_number] = sample_predicted

  actual_list = []
  predicted_list = []
  for file_identifier in unique_file_identifiers:
    actual_list.append(actual[file_identifier])
    predicted_list.append(predicted[file_identifier])

  return evaluate(actual_list, predicted_list)

def print_evaluation(dataset_name, evaluation):
  print(f'\n--- Dataset: {dataset_name}')
  print(f'Mean Average Precision: {format_number(evaluation["mean_avg_precision"])}')
  print(f'Mean Reciprocal Rank: {format_number(evaluation["mean_reciprocal_rank"])}')
  print(f'Mean R-Precision: {format_number(evaluation["mean_r_precision"])}')
  print(f'Overall Precisions: { [format_number(num) for num in evaluation["overall_precisions"]]}')
  print('\n\n')

if __name__ == '__main__':
  measure_time('running baselines', run_baselines)
