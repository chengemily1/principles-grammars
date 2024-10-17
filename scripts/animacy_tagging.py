import wn as wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.wsd import lesk
from nltk.corpus import wordnet31 as wn
from nltk import sent_tokenize
from googletrans import Translator

from loader import UD_Dataset

TRANSLATOR = Translator()
ANIMATE_SYNSETS = [synset for synset in list(wn.all_synsets('n')) if synset.lexname() in 
    ['noun.animal', 'noun.person']
]
SYNSETS = [synset for synset in list(wn.all_synsets('n'))]
ANIMATE_SYNSETS = [synset for synset in SYNSETS if synset.lexname() in ['noun.animal', 'noun.person']]


GERMAN_SYNSETS = wordnet.synsets(pos='n', lexicon='odenet:1.4')
GERMAN_WORDS = wordnet.words(pos='n', lexicon='odenet:1.4')

de_lemmas_file = open('/path/to/german_animate_lemmas_from_english.txt', 'r')
GERMAN_ANIMATE_LEMMAS = de_lemmas_file.read().split('\n')
de_lemmas_file.close()

def get_all_lemmas(lang):
    # Special case for German as it's not integrated in nltk
    if lang not in ('de', 'deu'):
        return set([n.lower() for synset in SYNSETS for n in synset.lemma_names(lang)])
    else:
        return set([n.lemma().lower().replace(' ', '_') for n in GERMAN_WORDS])

def get_animate_lemmas(lang):
    """Returns all animate lemmas in the target language.

    Args:
        lang (str): One of {fra, ita, cat, spa}

    Returns:
        set: set of animate lemmas in the target language
    """
    if lang not in ('de', 'deu'):
        return set([n.lower() for synset in ANIMATE_SYNSETS for n in synset.lemma_names(lang)])
    else:        
        return GERMAN_ANIMATE_LEMMAS


if __name__=='__main__':     
    pass