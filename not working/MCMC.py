#!/usr/bin/env python
import optparse
import sys
import models
import random
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--samples-per-sentence", dest="sps", default=1000, type="int", help="Number of MCMC samples per sentence (default=1000)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))

def mcmc_decode(f):
    current_translation = [random.choice(tm[(word,)]).english for word in f]
    current_logprob = compute_logprob(current_translation)
    
    best_translation = current_translation
    best_logprob = current_logprob
    
    for _ in range(opts.sps):
        # Propose a new translation by randomly changing one phrase
        new_translation = current_translation[:]
        idx = random.randint(0, len(f) - 1)
        new_translation[idx] = random.choice(tm[(f[idx],)]).english
        
        new_logprob = compute_logprob(new_translation)
        
        # Accept or reject the new translation based on Metropolis-Hastings criterion
        if np.log(random.random()) < new_logprob - current_logprob:
            current_translation = new_translation
            current_logprob = new_logprob
            
            if current_logprob > best_logprob:
                best_translation = current_translation
                best_logprob = current_logprob
    
    return " ".join(best_translation)

def compute_logprob(translation):
    logprob = 0.0
    lm_state = lm.begin()
    for word in translation:
        lm_state, word_logprob = lm.score(lm_state, word)
        logprob += word_logprob
    logprob += lm.end(lm_state)
    return logprob

for f in french:
    best_translation = mcmc_decode(f)
    print(best_translation)

    if opts.verbose:
        sys.stderr.write("Best Translation: %s\n" % best_translation)