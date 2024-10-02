#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input",
                     help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                     help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                     help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents",
                     default=sys.maxsize, type="int",
                     help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k",
                     default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose",
                     action="store_true", default=False,
                     help="Verbose mode (default=off)")

opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split())
          for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 0.0
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s with Greedy Search...\n" % (opts.input,))
for f in french:
    hypothesis = namedtuple(
        "hypothesis", "logprob, lm_state, predecessor, phrase, position")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0)
    h = initial_hypothesis

    while h.position < len(f):
        best_hypothesis = None
        best_logprob = -float('inf')
        i = h.position
        for j in range(i + 1, len(f) + 1):
            if f[i:j] in tm:
                for phrase in tm[f[i:j]]:
                    logprob = h.logprob + phrase.logprob
                    lm_state = h.lm_state
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob
                    new_hypothesis = hypothesis(
                        logprob, lm_state, h, phrase, j)
                    if logprob > best_logprob:
                        best_logprob = logprob
                        best_hypothesis = new_hypothesis
        if best_hypothesis is None:
            # Handle unknown words (should not happen due to tm augmentation)
            word = f[h.position]
            phrase = models.phrase(word, 0.0)
            logprob = h.logprob + phrase.logprob
            (lm_state, word_logprob) = lm.score(h.lm_state, word)
            logprob += word_logprob
            best_hypothesis = hypothesis(
                logprob, lm_state, h, phrase, h.position + 1)
        h = best_hypothesis

    # Extract English translation
    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (
            extract_english(h.predecessor), h.phrase.english)
    print(extract_english(h).strip())

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(h)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                         (h.logprob - tm_logprob, tm_logprob, h.logprob))