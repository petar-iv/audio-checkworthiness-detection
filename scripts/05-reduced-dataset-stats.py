import argparse

import numpy as np
import pandas as pd
from pydub import AudioSegment

from transformers import AutoTokenizer

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, filter_wav


def extract_reduced_dataset_stats():
  args = process_command_line_args()

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

  all_subsets_stats = []

  for subset in ['train', 'dev', 'test']:
    subset_stats = iterate_reduced_subset(args, tokenizer, subset)
    print_stats(subset, subset_stats)
    all_subsets_stats.append(subset_stats)
    print('\n\n')

  combined_stats = combine_stats(all_subsets_stats)
  print_stats('combined', combined_stats)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=str)
  parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-uncased')
  return parser.parse_args()

def iterate_reduced_subset(args, tokenizer, subset):
  subset_stats = {
    'events_count': 0,
    'sentences_count': 0,
    'checkworthy_count': 0,
    'not_checkworthy_count': 0,
    'text_token_counts': [],
    'audio_durations': []
  }
  iterate_texts_in_subset(args, tokenizer, subset, subset_stats)
  iterate_audio_files_in_subset(args, subset, subset_stats)
  return subset_stats

def iterate_texts_in_subset(args, tokenizer, subset, subset_stats):
  events = set()
  def extract_text_stats(event_name, event_path, file_name, file_path):
    events.add(event_name)

    df = pd.read_csv(file_path, sep='\t')
    for _, entry in df.iterrows():
      subset_stats['sentences_count'] += 1
      if entry['is_claim'] == 1:
        subset_stats['checkworthy_count'] += 1
      else:
        subset_stats['not_checkworthy_count'] += 1

      sentence = entry['sentence']
      tokens = tokenizer.encode_plus(sentence, truncation=False).input_ids
      subset_stats['text_token_counts'].append(len(tokens))

  iterate_reduced_event_files(
    reduced_dataset_dir(args.data_dir, 'v1') / '06-sentence-level-alignment' / subset,
    extract_text_stats,
    filter_tsv
  )

  subset_stats['events_count'] += len(events)

def iterate_audio_files_in_subset(args, subset, subset_stats):
  def extract_audio_stats(event_name, event_path, file_name, file_path):
    audio = AudioSegment.from_file(file_path)
    subset_stats['audio_durations'].append(audio.duration_seconds)

  iterate_reduced_event_files(
    reduced_dataset_dir(args.data_dir, 'v1') / '07-audio-segments' / subset,
    extract_audio_stats,
    filter_wav
  )

def combine_stats(all_subsets_stats):
  combined_stats = {
    'events_count': 0,
    'sentences_count': 0,
    'checkworthy_count': 0,
    'not_checkworthy_count': 0,
    'text_token_counts': [],
    'audio_durations': []
  }

  for subset_stats in all_subsets_stats:
    combined_stats['events_count'] += subset_stats['events_count']
    combined_stats['sentences_count'] += subset_stats['sentences_count']
    combined_stats['checkworthy_count'] += subset_stats['checkworthy_count']
    combined_stats['not_checkworthy_count'] += subset_stats['not_checkworthy_count']
    combined_stats['text_token_counts'].extend(subset_stats['text_token_counts'])
    combined_stats['audio_durations'].extend(subset_stats['audio_durations'])

  return combined_stats

def print_stats(subset, stats):
  print(f'-------------- Stats for {subset} --------------')
  print(f'Number of events: {stats["events_count"]}')
  print(f'Number of checkworthy sentences {stats["checkworthy_count"]}')
  print(f'Number of not checkworthy sentences {stats["not_checkworthy_count"]}')
  print(f'Number sentences {stats["sentences_count"]}')
  print('\nText token count stats:')
  print_sequences_stats(stats['text_token_counts'], [1, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800])
  print('\nAudio duration stats (in seconds):')
  print_sequences_stats(stats['audio_durations'], [1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 60, 120, 180, 240, 300, 360, 420, 480])
  print(f'------------------------------------------------')

def print_sequences_stats(data, bins):
  print(f'Min: {"{:.2f}".format(np.min(data))}')
  print(f'Max: {"{:.2f}".format(np.max(data))}')
  print(f'Mean: {"{:.2f}".format(np.mean(data))}')
  print(f'Standard deviation: {"{:.2f}".format(np.std(data))}')
  print(f'95 percentile: {"{:.2f}".format(np.percentile(data, 95))}')
  print(f'90 percentile: {"{:.2f}".format(np.percentile(data, 90))}')
  print(f'85 percentile: {"{:.2f}".format(np.percentile(data, 85))}')
  print(f'80 percentile: {"{:.2f}".format(np.percentile(data, 80))}')

  count, bins = np.histogram(data, bins=bins)

  for i in range(0, len(bins) - 1, 1):
    left_boundary = bins[i]
    right_boundary = bins[i+1]
    print(f'{str(left_boundary).ljust(4)} : {str(right_boundary).ljust(4)} -> {count[i]}')

if __name__ == '__main__':
  measure_time('extracting stats', extract_reduced_dataset_stats)
