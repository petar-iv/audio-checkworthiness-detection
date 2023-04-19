import argparse
from pathlib import Path

import openl3
import numpy as np
import soundfile as sf

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_wav, filename_without_extension


def extract_openl3_embeddings():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    measure_time(f'processing {subset}', lambda: extract_embeddings_for_subset(args, subset))

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--segments_dir_name', required=True, type=str)
  parser.add_argument('--target_dir_name', required=True, type=str)
  return parser.parse_args()

def extract_embeddings_for_subset(args, subset):
  source_dir = reduced_dataset_dir(args.data_dir) / args.segments_dir_name / subset
  target_dir = reduced_dataset_dir(args.data_dir) / args.target_dir_name / subset

  iterate_reduced_event_files(source_dir, lambda event_name, event_dir, file_name, file_path: extract_embedding_for_file(file_path, target_dir / event_name), fn_file_filter=filter_wav, should_measure_time=True)

def extract_embedding_for_file(file_path, target_dir):
  target_dir.mkdir(parents=True, exist_ok=True)

  audio, sampling_rate = sf.read(file_path)
  embedding, _ = openl3.get_audio_embedding(audio, sampling_rate, content_type='env', embedding_size=512)

  np.save(target_dir / f'{filename_without_extension(file_path)}.npy', embedding)

if __name__ == '__main__':
  measure_time('extracting openl3 embeddings', extract_openl3_embeddings)
