Research using audio for detecting checkworthy sentences
========================================================

Research initiated at Sofia University (Bulgaria), Faculty of Mathematics and Informatics.
Goal: detecting checkworthy sentences in political events (debates, interviews and speeches)
using their audio.

:warning: This project will serve as a diploma thesis of a student at Sofia University.
At the moment of sharing this, the diploma thesis has not been yet submitted.
Please do not disclose any content.

Instructions:

1. Create a directory for this project and create a virtual environment in it (assuming Python 3 is used):

```sh
mkdir checkworthy-research && cd checkworthy-research
python3 -m venv .
source ./bin/activate
```

2. Clone this git repository and `cd` into it

3. Install dependencies

```sh
pip install -r requirements.txt
```

4. Export the project root as environment variable:

```sh
export PROJECT_ROOT=/<...>/checkworthy-research/using-audio-for-detecting-checkworthy-sentences
```

5. `cd` into `scripts` directory

6. Download the full transcripts ([CLEF2021, Check That! challenge](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/ec6370e19f67f63772aff963cd1ee48284d7a599/task1), task 1b) via running:

```sh
python 01-download-full-transcripts.py --data_dir ${PROJECT_ROOT}/data
```

7. Retrieve the audio for the events (see the files in [this directory](/data/clef2021/task-1b-english/reduced/v1/01-event-links/)).

Put the audio files in <project-root>/data/clef2021/task-1b-english/reduced/v1/02-audios.
Expected directory structure:

```
02-audios
|- dev
   |- 20170228_Trump_Congress_joint_session
      |- audio-1.wav
   | ...
   | ...
|- test
   |- 20170512_Trump_NBC_holt_interview
      |- audio-1.wav
   | ...
   | ...
|- train
   | ...
   | ...
   |- 20160907_NBC_commander_in_chief_forum
      |- audio-1.wav
      |- audio-2.wav
   | ...
   | ...
```

**Note**: Not all of the event recordings were found and some are incomplete. Hence, we have 2 datasets:
- The original CLEF2021, Check That! dataset - called _full_ or _original_.
- The dataset with audio data (which is a subset of the _full_ dataset) - called _reduced_.

8. Apply begin-end audio alignment

This is required as audio files not always cover all the lines of a transcript -
in some cases the audio may start at line 50 of a transcript (for example),
in other cases - it may not cover the whole transcript until the end.

With that step we align the audio files with their transcripts so they overlap.

The begin-end audio alignments could be found [here](/data/clef2021/task-1b-english/reduced/v1/03-begin-end-alignment/).

Run the following script to align the audio files and the transcripts:

```sh
python 02-apply-begin-end-manual-alignment.py --data_dir ${PROJECT_ROOT}/data
```

It takes about 9 minutes on a MacBook Pro with an Intel Core i9-9880H processor and 32GB of RAM.

9. Word-level text-audio alignment

No action required, it is already built.
The used audio-text aligner is [_Gentle_](https://github.com/lowerquality/gentle).
The word-level alignment can be found [here](/data/clef2021/task-1b-english/reduced/v1/05-word-level-alignment/).

10. Sentence-level text-audio alignment

The data is annotated on sentence level.
We have 2 classes - _0_ (sentence is not checkworthy) and _1_ (sentence is checkworthy).
We need to leverage the word-level alignment to build text-audio alignment on sentence level.
Run the following:

```sh
python 03-build-sentence-level-alignment.py --data_dir ${PROJECT_ROOT}/data > output-building-sentence-level-alignment.txt
```

Remarks:
- The script runs quickly (for about a minute on a Mac), but its output is verbose so it is handy to redirect it to a file if one would like to inspect it.
- The line numbers in the output are those from the full dataset, the original line numbers.
- The speaker of every sentence in the dataset is known. We filter out those with speaker 'SYSTEM', reason: there are several such entries that are marked as checkworthy although they are not (like applauses).
- If the end of a sentence is not found, then we traverse up to 3 words from the next sentence to find an ending of the previous one.
- Senetence-level segments that have a duration of less than a second are skipped. Some models cannot process audio that is that short.

11. Cut sentence-level audio segments

Run the following:

```sh
python 04-cut-sentence-level-segments.py --data_dir ${PROJECT_ROOT}/data
```

Remarks:
- The script takes less than 5 minutes on a Mac.
- Each segment is resamples to 16 kHz, is converted to mono (single channel) and has sample width set to 16 bits.
- The audio segment file names contain a line number - this is the corresponding line number in the _reduced_ dataset.

12. Extracting reduced dataset stats

One may find stats about the full dataset in [this paper](https://arxiv.org/pdf/2109.12987.pdf), page 7 (Task 1B).

For convenience, a file with stats about the reduced dataset is also present in this repository and could be found [here](/reduced-dataset-stats.txt).

One may extract the stats using:

```sh
python 05-reduced-dataset-stats.py --data_dir ${PROJECT_ROOT}/data > reduced-dataset-stats.txt
```

Remarks:
- It takes about half a minute for the script to run.
- By default stats about number of tokens are extracted using `distilbert-base-uncased` tokenizer,
but that could be changed via command-line option `--tokenizer_name`.

13. Train text model on the reduced data

Run the following:

```sh
python train.py \
   --model_type text \
   --model_name distilbert-base-uncased \
   --output_dir ${PROJECT_ROOT}/trained-models/testing/text \
   --data_dir ${PROJECT_ROOT}/data \
   --sentence_level_alignment_dir_name 06-sentence-level-alignment \
   --max_seq_length 100 \
   --num_train_epochs 10 \
   --learning_rate 5e-5 \
   --warmup_proportion 0.1 \
   --train_batch_size 300 \
   --eval_batch_size 300 \
   --mixed_precision
```

The results:

Epoch best on dev - 1,
MAP(dev) = 0.2734,
MAP(test) = 0.3758

Epoch best on test - 2,
MAP(dev) = 0.2038,
MAP(test) = 0.3921

Training took ~33 minutes on a Tesla T4 GPU with 16GB memory.

14. Train audio model on the reduced data

Run the following:

```sh
python train.py \
   --model_type audio \
   --model_name 'ntu-spml/distilhubert' \
   --output_dir ${PROJECT_ROOT}/trained-models/testing/audio \
   --data_dir ${PROJECT_ROOT}/data \
   --sentence_level_alignment_dir_name 06-sentence-level-alignment \
   --audio_segments_dir_name 07-audio-segments \
   --max_seq_length 128000 \
   --num_train_epochs 10 \
   --learning_rate 5e-5 \
   --warmup_proportion 0.1 \
   --train_batch_size 35 \
   --eval_batch_size 35 \
   --mixed_precision
```

The results:

Epoch best on dev - 4,
MAP(dev) = 0.1146,
MAP(test) = 0.2289

Epoch best on test - 4

Training took ~9.5 hours on a Tesla T4 GPU with 16GB memory.

Remarks:
- MAP = Mean Average Precision, [article](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52).
- The higher the MAP score, the better.
- MAP scores are calculated on event level. In this implementation they are calculated on audio file level for the reduced dataset, but in dev and test all events have just one audio file so the 2 approaches lead to the same result.
