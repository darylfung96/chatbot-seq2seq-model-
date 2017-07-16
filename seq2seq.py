from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1:"EOS"}
        self.numWords = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.numWords
            self.word2count = 1
            self.index2word[self.numWords] = word
            self.numWords += 1
        else:
            self.word2count[word] += 1




def unicodeToAscii(string):
   return ''.join(c for c in unicodedata.normalize('NFD', string)
            if unicodedata.category(c) != 'Mn')

# trim , lowercase and remove non-letter characters
def normalizeString(string):
    string = unicodeToAscii(string)
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    return string



# the data is english (tab) french, so we need to separate english and french here
# reverse = french to english. The other way around.
# it's like feeding the output to input, and input to output
def readLanguages(language1, language2, reverse=False):

    print("Reading lines from the languages")

    # read the data and split on new line. remove whitespaces too
    lines = open('data/%s-%s.txt' % (language1, language2), encoding='utf-8').\
        read().strip().split('\n')


    # since we have english (tab) french. we split them into pairs
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]


    if not reverse:
        input_language = Lang(language1)
        output_language = Lang(language2)
    else:
        pairs = [list(reversed(p)) for p in pairs]
        input_language = Lang(language2)
        output_language = Lang(language1)

    return input_language, output_language, pairs




MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def isPairMeetCriteria(p, reverse):
    englishIndex = 0
    otherLangIndex = 1
    if reverse:
        englishIndex = 1
        otherLangIndex = 0

    return len(p[englishIndex].split(' ')) < MAX_LENGTH and \
        len(p[otherLangIndex].split(' ')) < MAX_LENGTH and \
        p[englishIndex].startswith(eng_prefixes)


def filterPairs(pairs, reverse):
    return [pair for pair in pairs if isPairMeetCriteria(pair, reverse)]

# english language should always be the first argument
def dataPreparation(lang1, lang2, reverse=False):
    inputs, output, pairs = readLanguages(lang1, lang2, reverse)
    print("Read %s of pairs" % len(pairs))
    pairs = filterPairs(pairs, reverse)

    print("Reduced to only %s number of pairs" % len(pairs))

    for pair in pairs:
        inputs.addSentence(pair[0])
        output.addSentence(pair[1])

    print("{} has {} of words" .format(inputs.name, inputs.numWords))
    print("{} has {} of words" .format(output.name, output.numWords))

    return inputs, output, pairs

inputs, outputs, pairs = dataPreparation('eng', 'fra', True)
print(random.choice(pairs))



