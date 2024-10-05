There are two python programs here (-h for usage):

- `decode` translates input sentences from French to English.
- `grade` computes the model score of a translated sentence.

These commands work in a pipeline. For example:

    > python decode | python compute-model-score

There is also a module:

- `model.py` implements very simple interfaces for language models
 and translation models, so you don't have to :). 

You can finish the assignment without modifying this file at all. 
You should look at it if you need to understand the interface
to the translation and language model.

The `data` directory contains files derived from the Canadian Hansards,
originally aligned by Ulrich Germann:

- `input`: French sentences to translate.

- `tm`: a phrase-based translation model. Each line is in the form:

    French phrase ||| English phrase ||| log_10(translation_prob)

- `lm`: a trigram language model file in ARPA format.

    log_10(ngram_prob)   ngram   log_10(backoff_prob)

The language model and translation model are computed from the data 
in the align directory, using alignments from the Berkeley aligner.

## Baseline
python decode | python compute-model-score
TOTAL LM LOGPROB: -14.709625
TOTAL TM LOGPROB: 0.410658

Total corpus log probability (LM+TM): -1439.873990

## Beam Search

python decode-beam-search | python compute-model-score

TOTAL LM LOGPROB: -13.596412
TOTAL TM LOGPROB: 0.212423

Total corpus log probability (LM+TM): -1242.952264

Code file: `decode-beam-search`
Translations file: `translations` 

## Greedy Decoder

python decode-ext | python compute-model-score

TOTAL LM LOGPROB: -21.671028
TOTAL TM LOGPROB: 1.339583

Total corpus log probability (LM+TM): -1613.207835

Code file: `decode-ext`
Translations file: `translations_greedy`

NOTE: Please find our report in the `report.pdf` file.

### Team
Anushri Suresh - asures13@jh.edu
Suhas Sasetty - ssasett1@jh.edu
Yogeeshwar Selvaraj - yselvar1@jh.edu
