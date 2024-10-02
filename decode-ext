#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import itertools

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=6, type="int", help="Distortion limit (default=6)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, end")

def beam_search_decode(f):
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, tuple([False]*len(f)), 0)
    stacks = [{} for _ in range(len(f) + 1)]
    stacks[0][lm.begin(), tuple([False]*len(f))] = initial_hypothesis

    for i in range(len(f) + 1):
        for h in sorted(stacks[i].values(), key=lambda h: -h.logprob)[:opts.s]:
            for j in range(len(f)):
                if h.coverage[j]:
                    continue
                for k in range(j, min(len(f), j + opts.d)):
                    if f[j:k+1] in tm:
                        if k+1 - j > opts.d:
                            break
                        for phrase in tm[f[j:k+1]]:
                            logprob = h.logprob + phrase.logprob
                            lm_state = h.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            
                            new_coverage = list(h.coverage)
                            for l in range(j, k+1):
                                new_coverage[l] = True
                            new_coverage = tuple(new_coverage)
                            
                            logprob += lm.end(lm_state) if sum(new_coverage) == len(f) else 0.0
                            new_hypothesis = hypothesis(logprob, lm_state, h, phrase, new_coverage, k)
                            
                            key = (lm_state, new_coverage)
                            if key not in stacks[sum(new_coverage)] or stacks[sum(new_coverage)][key].logprob < logprob:
                                stacks[sum(new_coverage)][key] = new_hypothesis

    winner = max(stacks[-1].values(), key=lambda h: h.logprob)
    return winner

def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

for f in french:
    winner = beam_search_decode(f)
    print(extract_english(winner))

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))