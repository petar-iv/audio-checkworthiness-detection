import time
import datetime as dt
from os import listdir
from pathlib import Path
from os.path import abspath, splitext, basename


def full_dataset_dir(data_dir, version, subset=None):
  path = Path(data_dir) / 'clef2021' / 'task-1b-english' / 'full' / version
  return path if not subset else path / subset

def reduced_dataset_dir(data_dir, version):
  return Path(data_dir) / 'clef2021' / 'task-1b-english' / 'reduced' / version

def filename_without_extension(file_path):
  return splitext(basename(file_path))[0]

def list_dir(dir, fn_filter):
  content = listdir(dir)
  content.sort()
  return [item for item in content if fn_filter(item)]

def iterate_dir(dir, fn_action, fn_filter):
  for item in list_dir(dir, fn_filter):
    fn_action(dir, item, abspath(Path(dir) / item))

def iterate_full_event_files(events_dir, fn_file_action):
  iterate_dir(events_dir, fn_file_action, fn_filter=filter_tsv)

def iterate_reduced_event_files(events_dir, fn_file_action, fn_file_filter, should_measure_time=False):
  def process_single_event(event_name):
    iterate_dir(events_dir / event_name,
      lambda event_dir, file_name, file_path: fn_file_action(event_name, event_dir, file_name, file_path),
      fn_filter=fn_file_filter)

  for event_name in list_dir(events_dir, fn_filter=filter_event_name):
    if should_measure_time:
      measure_time(f'processing event {event_name}', lambda: process_single_event(event_name))
    else:
      process_single_event(event_name)

def filter_tsv(file_name):
  return file_name.endswith('.tsv')

def filter_event_name(file_name):
  return file_name.startswith('20') and '_' in file_name

def filter_wav(file_name):
  return file_name.endswith('.wav')

def audio_segment_file_name(audio_base_name, line_number):
  return f'{audio_base_name}-{str(line_number).zfill(4)}.wav'

def measure_time(description, fn):
  print(f'> {description} started at {dt.datetime.utcnow()}')
  start = time.time()
  fn()
  end = time.time()
  print(f'> ({dt.timedelta(seconds=end-start)}) {description} completed at {dt.datetime.utcnow()}')
