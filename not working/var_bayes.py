import numpy as np
import optparse
import sys
import models

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

class VariationalBayesDecoder:
    def __init__(self, tm, lm, french_sentence, max_iter=100, learning_rate=0.01):
        self.tm = tm
        self.lm = lm
        self.french = french_sentence
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.hidden_dim = len(self.french)  # Simplification: one hidden variable per French word
        self.alpha = np.random.rand(self.hidden_dim)  # Dirichlet prior
        self.beta = np.random.rand(len(self.french), self.hidden_dim)  # Latent variable probabilities

    def fit(self):
        for _ in range(self.max_iter):
            self.update_parameters()

    def update_parameters(self):
        for i, french_word in enumerate(self.french):
            expected_latent = self.compute_expected_latent(i, french_word)
            self.alpha += self.learning_rate * (expected_latent - self.alpha)
            self.beta[i] += self.learning_rate * (expected_latent - self.beta[i])

    def compute_expected_latent(self, idx, french_word):
        expected = np.zeros(self.hidden_dim)
        for j, phrase in enumerate(self.tm.get(french_word, [])):
            lm_prob = self.compute_lm_prob(phrase.english)
            expected[j] = phrase.logprob + lm_prob
        return np.exp(expected - np.max(expected))  # Softmax for numerical stability

    def compute_lm_prob(self, english_phrase):
        words = english_phrase.split()
        lm_state = self.lm.begin()
        logprob = 0.0
        for word in words:
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            logprob += word_logprob
        logprob += self.lm.end(lm_state)
        return logprob

    def decode(self):
        self.fit()
        translation = []
        for i, french_word in enumerate(self.french):
            best_idx = np.argmax(self.beta[i])
            best_phrase = self.tm.get(french_word, [models.phrase(french_word, 0.0)])[best_idx]
            translation.append(best_phrase.english)
        return " ".join(translation)

# Decoding
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    decoder = VariationalBayesDecoder(tm, lm, f)
    translation = decoder.decode()
    print(translation)

    if opts.verbose:
        tm_logprob = sum(phrase.logprob for phrase in decoder.tm.get(word, [models.phrase(word, 0.0)]) for word in f)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
            (decoder.compute_lm_prob(translation), tm_logprob, decoder.compute_lm_prob(translation) + tm_logprob))