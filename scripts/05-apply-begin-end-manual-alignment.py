import argparse
from pathlib import Path

import pandas as pd
from pydub import AudioSegment

from utils import measure_time, full_dataset_dir, reduced_dataset_dir, filename_without_extension


def apply_begin_end_alignment():
  args = process_command_line_args()

  target_dir = reduced_dataset_dir(args.data_dir) / '04-audio-level-pairs'

  for subset in ['train', 'dev', 'test']:
    df = load_begin_end_alignments(args, subset)
    create_audio_transcript_pairs_for_all_entries(args, subset, target_dir, df)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  return parser.parse_args()

def load_begin_end_alignments(args, subset):
  file_path = reduced_dataset_dir(args.data_dir) / '03-begin-end-alignment' / f'{subset}.tsv'
  return pd.read_csv(file_path, sep='\t')

def create_audio_transcript_pairs_for_all_entries(args, subset, target_dir, df):
  for _, entry in df.iterrows():
    measure_time(f'creating audio-transcript pair for {subset}/{entry["event"]}/{entry["audio_file"]}', lambda: create_audio_transcript_pair_for_single_entry(args, subset, target_dir, entry))
    print('\n')

def create_audio_transcript_pair_for_single_entry(args, subset, target_dir, entry):
  event = entry['event']
  audio_file = entry['audio_file']

  audio_file_path = reduced_dataset_dir(args.data_dir) / '02-audios' / subset / event / audio_file

  target_event_dir = target_dir / subset / event
  target_event_dir.mkdir(parents=True, exist_ok=True)

  cut_audio(audio_file_path, entry['audio_begin'], entry['audio_end'], target_event_dir, audio_file)

  transcript_file_path = full_dataset_dir(args.data_dir, subset) / f'{event}.tsv'
  cut_transcript(transcript_file_path, entry['transcript_begin'], entry['transcript_end'], target_event_dir, audio_file)

def cut_audio(audio_file_path, begin, end, tagret_dir, audio_file):
  audio = AudioSegment.from_file(audio_file_path)
  chunk_data = audio[begin * 1000 : end * 1000]
  chunk_data.export(tagret_dir / audio_file, format='wav')

def cut_transcript(transcript_file_path, begin, end, tagret_dir, audio_file):
  df = pd.read_csv(transcript_file_path, sep='\t', header=None, names=['line_number', 'speaker', 'sentence', 'is_claim'])
  df = df[(df['line_number'] >= begin) & (df['line_number'] <= end)]
  df = df.rename(columns={'line_number': 'original_line_number'})

  df.to_csv(tagret_dir / f'{filename_without_extension(audio_file)}.tsv', sep='\t', index=False)

if __name__ == '__main__':
  measure_time('applying begin-end manual alignment', apply_begin_end_alignment)
