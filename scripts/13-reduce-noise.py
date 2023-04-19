import os
import argparse
from pathlib import Path

import torch
import torchaudio
import noisereduce as nr
from pydub import AudioSegment

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_wav, adjust_segment


def reduce_noise():
  args = process_command_line_args()
  source_dir = reduced_dataset_dir(args.data_dir) / '08-audio-segments'
  target_dir = reduced_dataset_dir(args.data_dir) / '09-audio-segments-rn'

  reduce_noise_in_all_subsets(source_dir, target_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def reduce_noise_in_all_subsets(source_dir, target_dir):
  for subset in ['train', 'dev', 'test']:
    measure_time(f'reducing noise in {subset}', lambda: reduce_audio_in_subset(source_dir / subset, target_dir / subset))

def reduce_audio_in_subset(source_subset_dir, target_subset_dir):
  def reduce_noise_in_segment(event_name, event_dir, file_name, file_path):
    audio, sample_rate = torchaudio.load(file_path)
    audio = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.95, n_jobs=-1)

    event_target_dir = target_subset_dir / event_name
    event_target_dir.mkdir(parents=True, exist_ok=True)

    target_file_path = event_target_dir / file_name

    torchaudio.save(target_file_path, torch.Tensor(audio), sample_rate)

    audio = AudioSegment.from_file(target_file_path)
    os.remove(target_file_path)
    audio = adjust_segment(audio)
    audio.export(target_file_path, format='wav')

  iterate_reduced_event_files(source_subset_dir, reduce_noise_in_segment, fn_file_filter=filter_wav, should_measure_time=True)

if __name__ == '__main__':
  measure_time('reducing noise', reduce_noise)
