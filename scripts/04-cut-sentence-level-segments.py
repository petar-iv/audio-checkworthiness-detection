import argparse
import pandas as pd
from pydub import AudioSegment


from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filter_tsv, \
  filename_without_extension, audio_segment_file_name


def cut_audio():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    print()
    measure_time(f'Processing subset {subset}', lambda: cut_audio_for_all_events(args, subset))
  print()

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=str)
  return parser.parse_args()

def cut_audio_for_all_events(args, subset):
  events_dir = reduced_dataset_dir(args.data_dir, 'v1') / '06-sentence-level-alignment' / subset
  target_dir = reduced_dataset_dir(args.data_dir, 'v1') / '07-audio-segments' / subset

  def process_single_event_file(event_name, event_path, file_name, file_path):
    event_segments_dir = target_dir / event_name
    event_segments_dir.mkdir(parents=True, exist_ok=True)
    build_sentence_alignment_for_audio_file(args, event_name, file_path, event_segments_dir, subset)

  iterate_reduced_event_files(events_dir, process_single_event_file, fn_file_filter=filter_tsv, should_measure_time=True)


def build_sentence_alignment_for_audio_file(args, event, sentence_level_alignment_file_path, event_segments_dir, subset):
  audio_base_name = filename_without_extension(sentence_level_alignment_file_path)
  audio_file_path = reduced_dataset_dir(args.data_dir, 'v1') / '04-audio-level-pairs' / subset / event / f'{audio_base_name}.wav'
  audio = AudioSegment.from_file(audio_file_path)

  sentence_alignment_df = pd.read_csv(sentence_level_alignment_file_path, sep='\t')

  for _, entry in sentence_alignment_df.iterrows():
    line_number = entry['line_number']
    begin = entry['begin']
    end = entry['end']

    segment = audio[begin * 1000 : end * 1000]
    segment = adjust_segment(segment)

    assert line_number < 10000, f'line number {line_number} encountered'

    segment.export(event_segments_dir / audio_segment_file_name(audio_base_name, line_number), format='wav')

def adjust_segment(segment):
  segment = segment.set_frame_rate(16000)
  segment = segment.set_channels(1)
  segment = segment.set_sample_width(2)
  return segment

if __name__ == '__main__':
  measure_time('cut audio segments for all events', cut_audio)
