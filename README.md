Audio Checkworthiness Detection
===============================

Research on using audio data in detecting checkworthy claims in political events
(debates, interviews, and speeches).
Neural models using audio or both text and audio are prepared.
The results show that an audio model could boost the performance of a powerful textual one when combined.

The research was held at Sofia University (Bulgaria), [Faculty of Mathematics and Informatics](https://www.fmi.uni-sofia.bg/en).

## Paper

Paper describing this work can be found:

- https://ieeexplore.ieee.org/document/10447064 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- https://arxiv.org/pdf/2306.05535.pdf

## Short Description of the Project

What this work contains in short:
- [Multimodal dataset](/script-outputs/15-reduced-dataset-stats.txt) (text and audio in English) - 48 hours of speech,
34,489 sentences.
Labels are on sentence level - whether each contains a checkworthy claim or not.
- Variants of audio segments (audio files for each sentence):
  - [Original audio](/data/clef2021/task-1b-english/reduced/v1/01-event-links).
  - With reduced noise (using [noisereduce](https://pypi.org/project/noisereduce/)).
  - With generated speech (using [FastSpeech 2](https://arxiv.org/pdf/2006.04558.pdf)).
- Variant of the dataset with a single speaker (Donald Trump, see [checkworthy-by-speaker](/script-outputs/checkworthy-by-speaker)).
- Real-world data - the multimodal dataset is based on the one from the CLEF Check-That! 2021 Challenge (see [this paper](https://arxiv.org/pdf/2109.12987.pdf), page 7 (Task 1B) and the [event links](/data/clef2021/task-1b-english/reduced/v1/01-event-links)).
- Addressing class skewness of the train dataset via oversampling (duplicating checkworthy claims 15 and 30 times (referred to as '15x' ([example](/training-results/features/early-fusion/bert-and-hubert/best-train-15x)) and
'30x' ([example](/training-results/features/early-fusion/bert-and-hubert/best-train-30x)))) and
undersampling (removing non-checkworthy samples until they equalise with the checkworthy, referred to as '1:1' or 'train-balanced-1' ([example](/training-results/features/early-fusion/bert-and-hubert/best-train-balanced-1))).
The variant without changes to the train dataset is referred to as 'Without changes' or 'as-is' ([example](/training-results/features/early-fusion/bert-and-hubert/best-as-is)).
- Textual baselines:
  - N-gram baseline from the CLEF Check-That! 2021 Challenge ([reference](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/blob/ec6370e19f67f63772aff963cd1ee48284d7a599/task1/baselines/subtask_1b.py)).
  - Counts of named entities from [different categories](/script-outputs/24-ner.txt) (feedforward neural network).
  - BERT-base uncased ([paper](https://arxiv.org/pdf/1810.04805.pdf), [model](https://huggingface.co/bert-base-uncased)).
- Fine-tuning audio models:
  - wav2vec 2.0 ([paper](https://arxiv.org/pdf/2006.11477.pdf), [model](https://huggingface.co/facebook/wav2vec2-base-960h)).
  - HuBERT ([paper](https://arxiv.org/pdf/2106.07447.pdf), [model](https://huggingface.co/facebook/hubert-base-ls960)).
  - data2vec-audio ([paper](https://arxiv.org/pdf/2202.03555.pdf), [model](https://huggingface.co/facebook/data2vec-audio-base-960h)).
  - WavLM ([paper](https://arxiv.org/pdf/2110.13900.pdf), [model](https://huggingface.co/microsoft/wavlm-base)).
- Knowledge alignment - training an audio model to represent the input it receives in the same way
a fine-tuned textual model represents its input in a teacher-student mode.
- Training models on audio features:
  - MFCC (extracted with [openSMILE](https://audeering.github.io/opensmile/get-started.html#mfcc-features), GRU and Transformer Encoder).
  - L<sup>3</sup>-Net  (extracted with [openl3](https://pypi.org/project/openl3/), GRU and Transformer Encoder).
  - i-vector (extracted with [kaldi](https://kaldi-asr.org/), feedforward neural network).
  - Interspeech ComParE 2013 (extracted with [openSMILE](https://audeering.github.io/opensmile/get-started.html#the-interspeech-2013-compare-challenge-feature-set), feedforward neural network).
  - Interspeech ComParE 2016 (extracted with [openSMILE](https://audeering.github.io/opensmile-python/api/opensmile.FeatureSet.html#opensmile.FeatureSet.ComParE_2016), feedforward neural network).

## Metric

Mean Average Precision (MAP) is being used - the same metric as in CLEF Check-That! 2021 Challenge ([paper](https://arxiv.org/pdf/2109.12987.pdf)).
The higher the MAP score, the better.

## Best Results

This section contains the results from the textual baselines and most notable results using audio.
The best epoch is chosen according to the MAP score on the dev dataset
and the model at this epoch is used for evaluation over the test dataset.
The tables below are sorted according to MAP(test) in descending order.

### Results With Multiple Speakers

| Row \# | Model Type | Model | Train dataset variant | Audio segments variant | MAP(test) |
| ------ | ---------- | ----- | --------------------- | ---------------------- | --------- |
| 1 | Early fusion ensemble | BERT & HuBERT (rows 5 & 9) | Without changes | Original | [0.3817](/training-results/features/early-fusion/bert-and-hubert/best-as-is/result.txt) |
| 2 | Late fusion ensemble | BERT & HuBERT (rows 5 & 9) | 15x | Original | [0.3758](/training-results/features/late-fusion/bert-and-hubert/best-train-15x/result.txt) |
| 3 | Early fusion ensemble | BERT & aligned data2vec (rows 5 & 6) | Without changes | Original | [0.3735](/training-results/features/aligned-early-fusion/bert-and-data2vec/best-as-is/result.txt) |
| 4 | Late fusion ensemble | BERT & aligned data2vec (rows 5 & 6) | 30x | Original | [0.3724](/training-results/features/aligned-late-fusion/bert-and-data2vec/best-train-30x/result.txt) |
| 5 | Textual | BERT | 1:1 | N/A | [0.3715](/training-results/text/bert/bert-reduced-train-balanced-1/result.txt) |
| 6 | Aligned audio model | data2vec-audio | Without changes | Original | [0.2999](/training-results/alignment/data2vec/test-maps.txt) (best epoch on dev: [8](/training-results/alignment/data2vec/dev-maps.txt)) |
| 7 | Aligned audio model | wav2vec 2.0 | Without changes | Original | [0.2996](/training-results/alignment/wav2vec2/test-maps.txt) (best epoch on dev: [10](/training-results/alignment/wav2vec2/dev-maps.txt)) |
| 8 | Aligned audio model | HuBERT | Without changes | Original | [0.2787](/training-results/alignment/hubert/test-maps.txt) best epoch on dev: [12](/training-results/alignment/hubert/dev-maps.txt) |
| 9 | Audio | HuBERT | 30x | Original | [0.2526](/training-results/audio/hubert/best-train-30x/result.txt) |
| 10 | Textual | n-gram baseline | 15x | N/A | [0.2392](/script-outputs/run-baselines-reduced-seed-42-train-15x.txt#L34) |
| 11 | Audio | wav2vec 2.0 | 15x | Original | [0.2365](/training-results/audio/wav2vec2/best-train-15x/result.txt) |
| 12 | Audio | data2vec-audio | 30x | Redused noise | [0.2330](/training-results/audio-rn/data2vec/best-train-30x/result.txt) |
| 13 | Textual | FNN with named entities | 15x | N/A | [0.2228](/training-results/text/ner/best-train-15x/result.txt) |

### Results With Single Speaker

| Row \# | Model Type | Model | Audio segments variant | MAP(test) |
| ------ | ---------- | ----- | ---------------------- | --------- |
| 1 | Audio | wav2vec 2.0 | Reduced noise | [0.3427](/training-results/audio-rn/wav2vec2/best-trump/result.txt) |
| 2 | Textual | BERT | N/A | [0.3267](/training-results/text/bert/bert-reduced-trump/result.txt) |
| 3 | Textual | n-gram baseline | N/A | [0.2693](/script-outputs/run-baselines-reduced-seed-42-trump.txt#L34) |
| 4 | Audio | HuBERT | Original | [0.2478](/training-results/audio/hubert/best-trump/result.txt) |
| 5 | Textual | FNN with named entities | N/A | [0.2193](/training-results/text/ner/best-trump/result.txt) |
| 6 | Audio | data2vec-audio | Reduced noise | [0.2129](/training-results/audio-rn/data2vec/best-trump/result.txt) |


## Setup the Project

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

The first time also download `en_core_web_sm` from spacy.

4. Export the project root as environment variable:

```sh
export PROJECT_ROOT=/<...>/checkworthy-research/audio-checkworthiness-detection
```

5. `cd` into `scripts` directory

6. Download the full transcripts ([CLEF2021, Check That! challenge](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/ec6370e19f67f63772aff963cd1ee48284d7a599/task1), task 1b) via running:

```sh
python 01-download-full-transcripts.py --data_dir ${PROJECT_ROOT}/data
```

Sentences with speaker `SYSTEM` are not part of the multimodal datataset - there are several entries that are marked as checkworthy although they are not (like applauses).
One may check which these are via running:

```sh
python 02-extract-system-sentences.py --data_dir ${PROJECT_ROOT}/data
```

and having a look at the files in `data/clef2021/task-1b-english/full/v1/system-sentences`.

7. Retrieve the audio for the events (see [event links](/data/clef2021/task-1b-english/reduced/v1/01-event-links/))

Put the audio files in ${PROJECT_ROOT}/data/clef2021/task-1b-english/reduced/v1/02-audios.
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
- The original CLEF Check-That! 2021 dataset - called _full_ or _original_.
- The multimodal dataset (which is a subset of the _full_ dataset) - also called _reduced_.

8. Apply begin-end audio alignment

This is required as audio files not always cover all the lines of a transcript -
in some cases the audio may start at line 50 of a transcript (for example),
in other cases - it may not cover the whole transcript until the end.

With that step we align the audio files with their transcripts.
The begin-end audio alignments could be found [here](/data/clef2021/task-1b-english/reduced/v1/03-begin-end-alignment/).
Run the following script to align the audio files and the transcripts:

```sh
python 05-apply-begin-end-manual-alignment.py --data_dir ${PROJECT_ROOT}/data
```

It takes about [8 minutes](/script-outputs/05-apply-begin-end-manual-alignment.txt) on a MacBook Pro with an Intel Core i9-9880H processor and 32GB of RAM.

9. Word-level text-audio alignment

No action required, it is already built.
The audio-text aligner in use is [Gentle](https://github.com/lowerquality/gentle).
The word-level alignment is available [here](/data/clef2021/task-1b-english/reduced/v1/05-word-level-alignment/).

10. Sentence-level text-audio alignment

The data is annotated on sentence level.
We have 2 classes - _0_ (sentence is not checkworthy) and _1_ (sentence is checkworthy).
We need to leverage the word-level alignment to build text-audio alignment on sentence level.
Run the following:

```sh
python 10-build-sentence-level-alignment.py --data_dir ${PROJECT_ROOT}/data > output-building-sentence-level-alignment.txt
```

Remarks:
- The script runs quickly (for about [a minute](/script-outputs/10-build-sentence-level-alignment.txt)),
but its output is verbose so it is handy to redirect it to a file in case one would like to inspect it.
- The line numbers in the output are those from the _full_ dataset, the original line numbers.
- We filter out sentences with speaker `SYSTEM`.
- If the end of a sentence is not found, then we traverse up to 3 words from the next sentence to find an ending of the previous one.
- Senetence-level segments that have a duration of less than a second are skipped. Some models cannot process audio that is that short.

11. Cut sentence-level audio segments

Run the following:

```sh
python 12-chop-audio-segments.py --data_dir ${PROJECT_ROOT}/data
```

Remarks:
- The script takes less than [5 minutes](/script-outputs/12-chop-audio-segments.txt) to run.
- Each segment is resampled to 16 kHz, is converted to mono (single channel) and has sample width set to 16 bits.
- The audio segment file names contain a line number - this is the corresponding line number in the _reduced_ dataset.

12. Reducing noise

This is performed using:

```sh
python 13-reduce-noise.py --data_dir ${PROJECT_ROOT}/data
```

This takes approximately [2 and a half hours](/script-outputs/13-reduce-noise.txt).

13. Generating speech

The results with generated speech were not the worst, but never the best.
As it consumes time, it is up to the reader to decide whether to invest into that approach.
Run the following to generate those audio segments:

```sh
python 14-generate-speech.py --data_dir ${PROJECT_ROOT}/data --target_dir_name ${PROJECT_ROOT}/data/clef2021/task-1b-english/reduced/v1/10-audio-segments-gs
```

[Output](/script-outputs/14-generate-speech.txt) from our run.

14. Extracting reduced dataset stats

One may find stats about the full dataset in [this paper](https://arxiv.org/pdf/2109.12987.pdf), page 7 (Task 1B).

For convenience, a file with stats about the reduced dataset is also present in this repository and could be found [here](/script-outputs/15-reduced-dataset-stats.txt).

One may extract the stats using:

```sh
python 15-reduced-dataset-stats.py --data_dir ${PROJECT_ROOT}/data > reduced-dataset-stats.txt
```

Remarks:
- It takes about half a minute for the script to run.
- By default stats about number of tokens are extracted using `distilbert-base-uncased` tokenizer,
but that could be changed via command-line option `--tokenizer_name`.

15. Extract MFCC, ComParE 2013, and ComParE 2016 features with openSMILE

Download openSMILE from [here](https://github.com/audeering/opensmile/releases).

Here is a template command to use:

```sh
python 16-build-openSMILE-features.py \
   --data_dir ${PROJECT_ROOT}/data \
   --segments_dir_name <segments-dir> \
   --target_dir_name <features-dir-name>/<features-type-dir-name> \
   --openSMILE_dir <path-to-opensmile>/opensmile-3.0 \
   --feature_set <feature-set>
```

It is executed for every variant of the audio segments (original, reduced noise, and generated speech)
and for every type of features - MFCC, ComParE 2013, and ComParE 2016.

And here are the values for the placeholders:

| Placeholder | Value |
| ----------- | ----- |
| \<segments-dir\> | For original audio use `08-audio-segments`.<br/>For reduced noise: `09-audio-segments-rn`.<br/>For generated speech: `10-audio-segments-gs`. |
| \<features-dir-name\> | For original audio use `features`.<br/>For reduced noise: `features-rn`.<br/>For generated speech: `features-gs`. |
| \<features-type-dir-name\> | For MFCC use `opensmile-mfcc`.<br/>For ComParE 2013: `compare-2013`.<br/>For ComParE 2016: `compare-2016`. |
| \<path-to-opensmile\> | The path where you have extracted openSMILE to, up to 'bin' (does not include it). |
| \<feature-set\> | For MFCC use `mfcc`.<br/>For ComParE 2013: `compare-2013`.<br/>For ComParE 2016: `compare-2016`. |

16. Extracting i-vectors

First, prepare Kaldi. See the instructions in [this script](/scripts/17-prepare-kaldi-asr.py#L11-L13)
and then run it.

Template for the next command to run (executed for every audio segments variant (original, reduced noise, and generated speech)):

```sh
python 18-extract-ivectors.py \
   --data_dir ${PROJECT_ROOT}/data \
   --segments_dir_name <segments-dir> \
   --target_dir_name <features-dir-name>/ivectors \
   --kaldi_dir <path-to-kaldi> \
   --mfcc_config ${PROJECT_ROOT}/kaldi-configs/mfcc.conf
```

The placeholders `<segments-dir>` and `<features-dir-name>` are analogous
to those in the openSMILE feature extraction.
`<path-to-kaldi>` is chosen by the reader as described in [this script](/scripts/17-prepare-kaldi-asr.py#L11-L13).

17. Extracting L<sup>3</sup>-Net features

Command template:

```sh
python 19-extract-openl3-embeddings.py \
   --data_dir ${PROJECT_ROOT}/data \
   --segments_dir_name <segments-dir> \
   --target_dir_name <features-dir-name>/openl3
```

The placeholders `<segments-dir>` and `<features-dir-name>` are analogous
to those in the openSMILE feature extraction.

If in doubt about the `--target_dir_name`, check where the features are read from afterwards
(during training) in [this script](/scripts/build-complete-reduced-data-files.py#L48-L76).

18. Oversampling

Run the following:

```sh
python 20-duplicate-train-checkworthy.py --data_dir ${PROJECT_ROOT}/data --num_duplicates 15
```

and then:

```sh
python 20-duplicate-train-checkworthy.py --data_dir ${PROJECT_ROOT}/data --num_duplicates 30
```

19. Undersampling

Run the following:

```sh
python 21-balance-dataset.py --data_dir ${PROJECT_ROOT}/data
```

20. Create the single speaker subset

Run the following:

```sh
python 22-filter-trump.py --data_dir ${PROJECT_ROOT}/data
```

21. Extract textual features - counts of named entities

Run the following:

```sh
python 24-ner.py --data_dir ${PROJECT_ROOT}/data
```

22. Generate the data files

All the features we have extracted so far are on sentence level
and are stored in files.
The final data files used during training contain path to those files with features.
Run the following commands:

```sh
python build-complete-reduced-data-files.py --data_dir ${PROJECT_ROOT}/data --train_dir_name train --dev_dir_name dev --test_dir_name test
```

```sh
python build-complete-reduced-data-files.py --data_dir ${PROJECT_ROOT}/data --train_dir_name train-15x --dev_dir_name dev --test_dir_name test
```

```sh
python build-complete-reduced-data-files.py --data_dir ${PROJECT_ROOT}/data --train_dir_name train-30x --dev_dir_name dev --test_dir_name test
```

```sh
python build-complete-reduced-data-files.py --data_dir ${PROJECT_ROOT}/data --train_dir_name train-balanced-1 --dev_dir_name dev --test_dir_name test
```

```sh
python build-complete-reduced-data-files.py --data_dir ${PROJECT_ROOT}/data --train_dir_name trump-train --dev_dir_name trump-dev --test_dir_name trump-test
```

23. Train scalers

In this step scalers for the different feature types are being prepared.
Run the following:

```sh
python train-scalers.py --data_dir ${PROJECT_ROOT}/data --target_dir ${PROJECT_ROOT}/trained-scalers
```

Output from our run could be found [here](/script-outputs/train-scalers.txt).

24. Train models

Run the [train.py](/scripts/train.py) script to train a model.
Refer to the experiments in [training-results](/training-results) for hyperparameter values.
For example, [these are the values](/training-results/text/bert/bert-reduced/args.json) used for fine-tuning BERT on the multimodal dataset (without changes to the train dataset for addressing skewness).
And [these are the values](/training-results/audio/hubert/best-as-is/args.json) when fine-tuning HuBERT on the same format of the train dataset (without changes, 'as-is').

Use the [alignment.py](/scripts/alignment.py) script for the knowledge alignment procedure in teacher-student mode. Note that the textual model should be fine-tuned in advance.

Use the [extract-vector-representations.py](/scripts/extract-vector-representations.py) script for extracting vector representations of already fine-tuned textual or audio models.

25. Inspect ranking performed by a model

Example command for inspecting actually checkworthy claims and their ranks as given by a model.
It also displays the top-ranked and bottom-ranked sentences.
Run the following:

```sh
python inspect-ranking.py \
   --data_dir ${PROJECT_ROOT}/data \
   --actual_dir_name test \
   --predictions_dir ${PROJECT_ROOT}/training-results/features/early-fusion/bert-and-hubert/best-as-is/epoch-2/test-predictions
```
