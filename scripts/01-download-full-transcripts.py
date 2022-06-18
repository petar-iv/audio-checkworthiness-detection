import zipfile
import argparse
from pathlib import Path
from shutil import move, rmtree
from urllib.request import urlretrieve

from utils import full_dataset_dir, iterate_full_event_files


base_url = 'https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/raw/ec6370e19f67f63772aff963cd1ee48284d7a599/task1'

def download_full_transcripts():
  args = process_command_line_args()
  version = 'v1'

  train_dir, dev_dir, test_dir = create_transcripts_dirs(args, version)

  download_train_and_dev_subsets(version, train_dir, dev_dir)
  download_test_subset(test_dir)

  print('Download of transcripts done')

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True, type=str)
  return parser.parse_args()

def create_transcripts_dirs(args, version):
  data_dir = Path(args.data_dir)
  version_dir = full_dataset_dir(data_dir, version)
  version_dir.mkdir(parents=True, exist_ok=True)

  train_dir = version_dir / 'train'
  train_dir.mkdir(parents=True, exist_ok=True)

  dev_dir = version_dir / 'dev'
  dev_dir.mkdir(parents=True, exist_ok=True)

  test_dir = version_dir / 'test'
  test_dir.mkdir(parents=True, exist_ok=True)

  return train_dir, dev_dir, test_dir

def download_train_and_dev_subsets(version, train_dir, dev_dir):
  def move_transcripts(archive_content_dir):
    move_files(Path(archive_content_dir) / version / 'train', train_dir)
    move_files(Path(archive_content_dir) / version / 'dev', dev_dir)

  download_and_process(f'{base_url}/data/subtask-1b--english/{version}.zip', move_transcripts)

def download_test_subset(test_dir):
  def move_transcripts(archived_content_dir):
    move_files(Path(archived_content_dir) / 'subtask-1b--english', test_dir)

  download_and_process(f'{base_url}/test-gold/subtask-1b--english.zip', move_transcripts)

def download_and_process(archive_url, fn_process):
  try:
    archive_file = 'archive.zip'
    urlretrieve(archive_url, archive_file)

    archive_content_dir = 'archive_content'
    with zipfile.ZipFile(archive_file, 'r') as zip:
      zip.extractall(archive_content_dir)

    fn_process(archive_content_dir)
  finally:
    rmtree(archive_content_dir)
    Path(archive_file).unlink()

def move_files(source_dir, target_dir):
  def move_single_file(source_subset_dir, source_file_name, source_file_path):
    move(source_file_path, target_dir)

  iterate_full_event_files(source_dir, move_single_file)

if __name__ == '__main__':
  download_full_transcripts()
