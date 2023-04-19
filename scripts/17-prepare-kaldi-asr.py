import tarfile
import argparse
from pathlib import Path
from urllib.request import urlretrieve


def prepare_kaldi():
  args = process_command_line_args()
  assert_no_spaces(str(args.kaldi_dir))

  print(f"Clone the Kaldi's git repo https://github.com/kaldi-asr/kaldi.git to {args.kaldi_dir}/repo")
  print('Commit which has been used in this project is e4940d045d39deb86016bc176893303b5240ff59, it is recommended to use the same one')
  print('See the instructions in the INSTALL file on root level in the repo on how to setup kaldi')

  download_pretrained_model(args.kaldi_dir)

def process_command_line_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--kaldi_dir', required=True, type=Path)
  return parser.parse_args()

def assert_no_spaces(str):
  if ' ' in str:
    raise Exception(f'Value "{str}" contains spaces, but should not')

def download_pretrained_model(kaldi_dir):
  model_dir = kaldi_dir / 'pretrained-model'
  model_dir.mkdir(parents=True, exist_ok=True)

  # See https://desh2608.github.io/2020-05-18-using-librispeech/
  download_and_unpack('http://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz', model_dir)
  download_and_unpack('http://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz', model_dir)

def download_and_unpack(archive_url, working_dir):
  try:
    archive_file = working_dir / 'archive.tar.gz'
    urlretrieve(archive_url, archive_file)

    with tarfile.open(archive_file) as tar:
      tar.extractall(working_dir)
  finally:
    archive_file.unlink()

if __name__ == '__main__':
  prepare_kaldi()
