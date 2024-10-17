from datasets import load_dataset
import re
import argparse
import json
import stanza
from stanza.utils.conll import CoNLL
from stanza.resources.common import load_resources_json
import typing
from tqdm import tqdm
import time
import multiprocessing as mp   
from multiprocessing import set_start_method

from annotate_data import *

NUMBER_OF_CHUNKS = 20 # can't process entire dataset at once.

def f(data_point, lang, nlp):
    return '# newdoc\n' + CoNLL.doc2conll_text(nlp(data_point))

if __name__=='__main__':
    set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
                    prog = 'Wiki to Conllu',
                    description = 'Downloads Wikipedia data from Huggingface and writes conllu to file.',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('--lang', type=str,
    choices=['zh', 'fr', 'es', 'ca', 'it', 'de', 'en', 'nl', 'sv', 'ru', 'pl', 'sl', 'mt', 'he', 'ar'])
    parser.add_argument('--chunk', type=float, choices=list(range(NUMBER_OF_CHUNKS)))
    parser.add_argument('--savepath', type=str)
    args = parser.parse_args()

    # Loads the wikipedia data from huggingface
    if args.lang in ['fr', 'it', 'de']:
        dataset_name = args.lang 
        data = load_dataset("wikipedia", dataset_name, beam_runner='DirectRunner')
    elif args.lang not in ['fr', 'it', 'de']:
        beam_runner, date = 'DirectRunner', '20230120'
        data = load_dataset("olm/wikipedia", language=args.lang, date=date)

    print('data loaded')
    
    train_set = []
    for i in range(len(data['train'])):
        try:
            train_set.append(data['train'][i]['text'])
        except:
            print('issue with: ', i) 

    # Only process the chunk
    print('Number of articles: ', len(train_set))
    chunk_size = len(train_set) // (NUMBER_OF_CHUNKS)
    print('chunk size: ', chunk_size)
    train_set = train_set[int(args.chunk) * chunk_size: (int(args.chunk) + 1) * chunk_size]

    # Filter processors depending on availability (eg mwt no exist in swedish)
    resources = set(load_resources_json()[args.lang].keys())
    processors = ','.join(set(['tokenize', 'mwt', 'pos', 'lemma']).intersection(resources))

    nlp = stanza.Pipeline(lang=args.lang, processors=processors, gpu=True)
    docs_as_conllu_text = [f(text, args.lang, nlp) for text in tqdm(train_set)]

    conllu_text = ''.join(docs_as_conllu_text)
    filename = args.savepath
    
    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(conllu_text)

