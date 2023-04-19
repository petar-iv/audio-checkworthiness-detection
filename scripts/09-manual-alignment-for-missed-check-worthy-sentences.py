import argparse
from pathlib import Path

from pydub import AudioSegment

from utils import measure_time, reduced_dataset_dir, filename_without_extension


def manual_alignment_of_missed_check_worthy_sentences():
  args = process_command_line_args()

  alignments = manual_alignment_on_train_set()
  apply_alignment(args, alignments, 'train')

  alignments = manual_alignment_on_dev_set()
  apply_alignment(args, alignments, 'dev')

  alignments = manual_alignment_on_test_set()
  apply_alignment(args, alignments, 'test')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=Path)
  parser.add_argument('--target_dir', required=True, type=Path)
  return parser.parse_args()

def manual_alignment_on_train_set():
  return [
    { 'event': '20160213_GOP_Greenville', 'original_line_number': 1002, 'audio_file': 'audio-1.wav', 'begin': 3749.5, 'end': 3752 },

    { 'event': '20160303_GOP_Michigan', 'original_line_number': 418, 'audio_file': 'audio-1.wav', 'begin': 1201.5, 'end': 1208 },

    { 'event': '20160926_first_presidential_debate', 'original_line_number': 183, 'audio_file': 'audio-1.wav', 'begin': 830.3, 'end': 831.7 },
    { 'event': '20160926_first_presidential_debate', 'original_line_number': 1058, 'audio_file': 'audio-1.wav', 'begin': 4512.6, 'end': 4513.8 },

    { 'event': '20161005_vice_presidential_debate', 'original_line_number': 236, 'audio_file': 'audio-1.wav', 'begin': 1105.2, 'end': 1108 },
    { 'event': '20161005_vice_presidential_debate', 'original_line_number': 305, 'audio_file': 'audio-1.wav', 'begin': 1297.8, 'end': 1301.4 },
    { 'event': '20161005_vice_presidential_debate', 'original_line_number': 968, 'audio_file': 'audio-1.wav', 'begin': 3862, 'end': 3864.5 },

    { 'event': '20161010_second_presidential_debate', 'original_line_number': 607, 'audio_file': 'audio-1.wav', 'begin': 2794.3, 'end': 2795.3 },
    { 'event': '20161010_second_presidential_debate', 'original_line_number': 838, 'audio_file': 'audio-1.wav', 'begin': 3708.8, 'end': 3709.9 },
    { 'event': '20161010_second_presidential_debate', 'original_line_number': 839, 'audio_file': 'audio-1.wav', 'begin': 3709.8, 'end': 3710.8 },

    { 'event': '20161019_third_presidential_debate', 'original_line_number': 345, 'audio_file': 'audio-1.wav', 'begin': 1631, 'end': 1632 },
    { 'event': '20161019_third_presidential_debate', 'original_line_number': 766, 'audio_file': 'audio-1.wav', 'begin': 3277.4, 'end': 3278.4 },
    { 'event': '20161019_third_presidential_debate', 'original_line_number': 1038, 'audio_file': 'audio-1.wav', 'begin': 4334.8, 'end': 4336 }
  ]

def manual_alignment_on_dev_set():
  return []

def manual_alignment_on_test_set():
  return [
    { 'event': '20180615_Trump_lawn', 'original_line_number': 44, 'audio_file': 'audio-1.wav', 'begin': 46.2, 'end': 47.3 },
    { 'event': '20180615_Trump_lawn', 'original_line_number': 279, 'audio_file': 'audio-1.wav', 'begin': 623.1, 'end': 624.2 },

    { 'event': '20180628_Trump_NorthDakota', 'original_line_number': 348, 'audio_file': 'audio-1.wav', 'begin': 1525.5, 'end': 1527 },
    { 'event': '20180628_Trump_NorthDakota', 'original_line_number': 509, 'audio_file': 'audio-1.wav', 'begin': 2287.5, 'end': 2288.5 }
  ]

def apply_alignment(args, alignments, subset):
  events_dir = reduced_dataset_dir(args.data_dir) / '04-audio-level-pairs' / subset
  for alignment in alignments:
    audio_file = events_dir / alignment['event'] / alignment['audio_file']
    target_dir = args.target_dir / subset / alignment['event']
    target_dir.mkdir(parents=True, exist_ok=True)
    cut_audio(audio_file, alignment['begin'], alignment['end'], target_dir, f"{filename_without_extension(alignment['audio_file'])}-{str(alignment['original_line_number']).zfill(4)}.wav")

def cut_audio(audio_file_path, begin, end, target_dir, file_name):
  audio = AudioSegment.from_file(audio_file_path)
  chunk_data = audio[begin * 1000 : end * 1000]
  chunk_data.export(target_dir / file_name, format='wav')

if __name__ == '__main__':
  measure_time('extracting missed checkworthy sentences', manual_alignment_of_missed_check_worthy_sentences)
