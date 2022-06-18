import re
import math
import argparse
import pandas as pd

from utils import measure_time, reduced_dataset_dir, iterate_reduced_event_files, filename_without_extension


def build_sentence_alignment():
  args = process_command_line_args()

  for subset in ['train', 'dev', 'test']:
    build_sentence_alignment_for_all_events(args, subset)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=str)
  return parser.parse_args()

def build_sentence_alignment_for_all_events(args, subset):
  events_dir = reduced_dataset_dir(args.data_dir, 'v1') / '05-word-level-alignment' / subset
  sentence_alignment_dir = reduced_dataset_dir(args.data_dir, 'v1') / '06-sentence-level-alignment' / subset

  def process_single_event_file(event_name, event_path, file_name, file_path):
    print(f'Processing {event_name}/{file_name}')
    event_target_dir = sentence_alignment_dir / event_name
    event_target_dir.mkdir(parents=True, exist_ok=True)
    build_sentence_alignment_for_audio_file(args, event_name, file_path, event_target_dir, subset)

  iterate_reduced_event_files(events_dir, process_single_event_file, fn_file_filter=lambda filename: filename.endswith('.csv'))

def build_sentence_alignment_for_audio_file(args, event, word_alignment_file_path, target_dir, subset):
  word_alignment_df = pd.read_csv(word_alignment_file_path, header=None)
  word_alignment_df_index = 0

  audio_base_name = filename_without_extension(word_alignment_file_path)
  transcript_file_path = reduced_dataset_dir(args.data_dir, 'v1') / '04-audio-level-pairs' / subset / event / f'{audio_base_name}.tsv'
  transcript_df = pd.read_csv(transcript_file_path, sep='\t')

  result = []
  def add_to_result(original_line_number, speaker, sentence, is_claim, begin, end):
    line_number = len(result) + 1
    result.append({
      'original_line_number': original_line_number,
      'line_number': line_number,
      'speaker': speaker,
      'sentence': sentence,
      'is_claim': is_claim,
      'begin': begin,
      'end': end
    })


  for _, entry in transcript_df.iterrows():
    speaker = entry['speaker']
    line_number = entry['original_line_number']

    sentence = entry['sentence'].strip()
    sentence_words = split_words(sentence)

    is_claim = entry['is_claim']

    sentence_begin = -1
    sentence_end = -1
    accumulated_words = []

    while accumulated_words != sentence_words:
      row = word_alignment_df.iloc[word_alignment_df_index]

      word = row[0]
      word_begin = row[2]
      word_end = row[3]

      accumulated_words.append(word)

      if sentence_begin < 0 and not math.isnan(word_begin):
        sentence_begin = word_begin

      if not math.isnan(word_end):
        sentence_end = word_end
      else:
        if len(sentence_words) == len(accumulated_words) and sentence_begin >= 0:
          print(f'[WARN-1] line #{line_number} (text: "... {sentence[-150:]}") - audio end of last word ("{word}") not determined, trying to get the start of 1 of the next 3 words (if possible)')
          for i in range(3):
            if (i + word_alignment_df_index) < len(word_alignment_df):
              next_row = word_alignment_df.iloc[(i + word_alignment_df_index)]
              if not math.isnan(next_row[2]):
                sentence_end = next_row[2]
                print(f'- line end found at start of (last word) + {i+1}\n')
                break
            else:
              break

      word_alignment_df_index += 1

    if speaker == 'SYSTEM':
      print(f'Skipping line #{line_number} (speaker is SYSTEM)')
      continue

    rule = manual_rule(subset, event, line_number, audio_base_name)
    if rule:
      print(f'Applying manual rule for {subset}/{event}/{line_number}/{audio_base_name}.wav')
      add_to_result(line_number, speaker, sentence, is_claim, rule['begin'], rule['end'])
      continue

    if sentence_begin >= 0 and sentence_end >= 0:
      if sentence_end - sentence_begin >= 1:
        add_to_result(line_number, speaker, sentence, is_claim, sentence_begin, sentence_end)
      else:
        print(f'Skipping line #{line_number} (less than a second segment)')
      continue

    # handle audio starting with an unrecognized word
    if sentence_begin < 0 and sentence_end >= 0:
      if len(result) == 0:
        sentence_begin = 0
        print(f'[WARN-2] line #{line_number}, defaulting line audio start at audio start')
        if sentence_end - sentence_begin >= 1:
          add_to_result(line_number, speaker, sentence, is_claim, sentence_begin, sentence_end)
        else:
          print(f'Skipping line #{line_number} (less than a second segment)')
        continue
      else:
        print(f'[WARN-3] line #{line_number} (text: "{sentence[:150]} ...") - could not determine audio start, will be removed\n')
        continue

    if sentence_begin >= 0 and sentence_end < 0:
      print(f'[WARN-4] line #{line_number} (text: "... {sentence[-150:]}") - could not determine audio end, will be removed\n')
      continue

    if sentence_begin < 0 and sentence_end < 0:
      print(f'[WARN-5] line #{line_number} (text: "... {sentence[-150:]}") - could not determine audio start and end, will be removed\n')
      continue

  result_df = pd.DataFrame(result)
  print(f'Original length: {len(transcript_df)}, length after sentence audio alignment: {len(result_df)}')

  result_df.to_csv(f'{target_dir}/{audio_base_name}.tsv', sep='\t', index=False)

def manual_rule(subset, event, original_line_number, audio_base_name):
  rules = [
    # train
    { 'subset': 'train', 'event': '20160213_GOP_Greenville', 'original_line_number': 1002, 'audio_file': 'audio-1.wav', 'begin': 3749.5, 'end': 3752 },

    { 'subset': 'train', 'event': '20160303_GOP_Michigan', 'original_line_number': 418, 'audio_file': 'audio-1.wav', 'begin': 1201.5, 'end': 1208 },

    { 'subset': 'train', 'event': '20160926_first_presidential_debate', 'original_line_number': 183, 'audio_file': 'audio-1.wav', 'begin': 830.3, 'end': 831.7 },
    { 'subset': 'train', 'event': '20160926_first_presidential_debate', 'original_line_number': 1058, 'audio_file': 'audio-1.wav', 'begin': 4512.6, 'end': 4513.8 },

    { 'subset': 'train', 'event': '20161005_vice_presidential_debate', 'original_line_number': 236, 'audio_file': 'audio-1.wav', 'begin': 1105.2, 'end': 1108 },
    { 'subset': 'train', 'event': '20161005_vice_presidential_debate', 'original_line_number': 305, 'audio_file': 'audio-1.wav', 'begin': 1297.8, 'end': 1301.4 },
    { 'subset': 'train', 'event': '20161005_vice_presidential_debate', 'original_line_number': 968, 'audio_file': 'audio-1.wav', 'begin': 3862, 'end': 3864.5 },

    { 'subset': 'train', 'event': '20161010_second_presidential_debate', 'original_line_number': 607, 'audio_file': 'audio-1.wav', 'begin': 2794.3, 'end': 2795.3 },
    { 'subset': 'train', 'event': '20161010_second_presidential_debate', 'original_line_number': 838, 'audio_file': 'audio-1.wav', 'begin': 3708.8, 'end': 3709.9 },
    { 'subset': 'train', 'event': '20161010_second_presidential_debate', 'original_line_number': 839, 'audio_file': 'audio-1.wav', 'begin': 3709.8, 'end': 3710.8 },

    { 'subset': 'train', 'event': '20161019_third_presidential_debate', 'original_line_number': 345, 'audio_file': 'audio-1.wav', 'begin': 1631, 'end': 1632 },
    { 'subset': 'train', 'event': '20161019_third_presidential_debate', 'original_line_number': 766, 'audio_file': 'audio-1.wav', 'begin': 3277.4, 'end': 3278.4 },
    { 'subset': 'train', 'event': '20161019_third_presidential_debate', 'original_line_number': 1038, 'audio_file': 'audio-1.wav', 'begin': 4334.8, 'end': 4336 },

    # test
    { 'subset': 'test', 'event': '20180615_Trump_lawn', 'original_line_number': 44, 'audio_file': 'audio-1.wav', 'begin': 46.2, 'end': 47.3 },
    { 'subset': 'test', 'event': '20180615_Trump_lawn', 'original_line_number': 279, 'audio_file': 'audio-1.wav', 'begin': 623.1, 'end': 624.2 },

    { 'subset': 'test', 'event': '20180628_Trump_NorthDakota', 'original_line_number': 348, 'audio_file': 'audio-1.wav', 'begin': 1525.5, 'end': 1527 },
    { 'subset': 'test', 'event': '20180628_Trump_NorthDakota', 'original_line_number': 509, 'audio_file': 'audio-1.wav', 'begin': 2287.5, 'end': 2288.5 }
  ]

  for rule in rules:
    if rule['subset'] == subset and rule['event'] == event and rule['original_line_number'] == original_line_number and rule['audio_file'] == f'{audio_base_name}.wav':
      return rule

  return None

def split_words(line):
  # see https://github.com/lowerquality/gentle/blob/0.10.1/gentle/metasentence.py#L41
  words = []
  for m in re.finditer(r'(\w|\’\w|\'\w)+', line, re.UNICODE):
    words.append(m.group())
  return words

if __name__ == '__main__':
  measure_time('building sentence alignments', build_sentence_alignment)
