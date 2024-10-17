from conllu import parse
from conllu.models import TokenList
from scipy.stats import entropy
import matplotlib.pyplot as plt
import functools
import tikzplotlib as tpl
from scipy import stats
import numpy as np
import seaborn as sns
import time 
import pandas as pd
import random
from tqdm import tqdm
import nltk

from animacy_tagging import *
from nltk.corpus import wordnet as wn
import os 
from hebrew import Hebrew
from IPython.display import clear_output

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import wn
for lex in wn.lexicons():
    print(f'{lex.id}:{lex.version}\t{lex.label}')

from constants import *

## SET THE LANGUAGE
lang = 'pl' # modify here
wn_lang = wordnet_mapping[lang]
POSSIBLE_GENDERS = gender_mapping[lang]
POSSIBLE_NUMBERS = number_mapping[lang]

## LOAD IN CONLLU FILES TO CREATE DF
NOUNS = get_all_lemmas(wn_lang)
if lang == 'he':
    NOUNS = {str(Hebrew(noun).text_only()).replace('!', '') for noun in NOUNS}

ANIMATE_NOUNS = get_animate_lemmas(wn_lang)
if lang == 'he':
    ANIMATE_NOUNS = {str(Hebrew(noun).text_only()).replace('!', '') for noun in ANIMATE_NOUNS}

fpaths = [
    f'/path/to/language.conllu' for i in range(20) 
    if os.path.exists(f'/path/to/language.conllu')
]

# CONSTRUCT LIST OF ALL TOKENS
all_tokens_df = []
for i, filepath in enumerate(fpaths):
    nouns = {'text':[], 'lemma': [], 
             'gender': [], 
             'number': [], 'animate': []}
    with open(filepath, 'r') as file:
        line = True
        counter = 0
        t = time.time()

        while line:
            line = file.readline()
            split_line = line.split('\t')
            
            if ('NOUN' in split_line): 

                counter += 1
                if time.time() - t > 1:
                    clear_output(wait=False)
                    print('nouns compiled: {}'.format(counter))
                    t = time.time()
                try:
                    text, lemma, morph_features = \
                        split_line[1].lower(), split_line[2].lower(), split_line[5].split('|')
                    number = [feat for feat in morph_features if 'Number' in feat][0][len('Number='):]
                    if number == 'Plur' and lang == 'nl': 
                        gender = ''
                    elif lang == 'ar':
                        gender = None
                    else:
                        gender = [feat for feat in morph_features if 'Gender' in feat][0][len('Gender='):] if lang != 'en' else ''
                except:
                    continue
                    
                if lemma.lower().replace(' ', '_') in NOUNS or lemma in NOUNS or text.lower().replace(' ', '_') in NOUNS or text in NOUNS:
                    if lang == 'he':
                        text = str(Hebrew(text).text_only()).replace('!', "")
                        lemma = str(Hebrew(lemma).text_only()).replace('!', "")

                nouns['text'].append(text)
                nouns['lemma'].append(lemma)
                nouns['gender'].append(gender)
                nouns['number'].append(number)
                nouns['animate'].append(
                    lemma.lower().replace(' ', '_') in ANIMATE_NOUNS or lemma in ANIMATE_NOUNS or text.lower().replace(' ', '_') in ANIMATE_NOUNS \
                    or text in ANIMATE_NOUNS
                )

        tokens_df = pd.DataFrame(nouns).groupby(['text', 'lemma', 
                                                 'gender', 
                                                 'number', 'animate']).size().reset_index(name='count')
        all_tokens_df.append(tokens_df)
        
        
all_tokens_df = pd.concat(all_tokens_df)

# Aggregate by counting them
all_tokens_df = all_tokens_df.groupby(['text', 'lemma', 'number', 'animate']).agg('sum').reset_index()
all_tokens_df = all_tokens_df[all_tokens_df['gender'].isin(POSSIBLE_GENDERS) & all_tokens_df['number'].isin(POSSIBLE_NUMBERS)]

# Do some post-processing, make sure things are alphanumeric and the lengths and frequencies are > 1.
all_tokens_df = all_tokens_df[all_tokens_df['lemma'].isin(NOUNS) \
                              & all_tokens_df['lemma'].str.isalnum() \
                              & all_tokens_df['text'].str.isalnum() \
                              & (all_tokens_df['text'].str.len() > 1) \
                              & (all_tokens_df['lemma'].str.len() > 1) \
                              & (all_tokens_df['count'] > 1)
                             ]

all_tokens_df.sort_values('count', ascending=False)
all_tokens_df.to_csv(all_tokens_df_format.format(lang, lang))


# CONSTRUCT LIST OF ANIMATE TOKENS
animate_tokens_df = all_tokens_df[all_tokens_df['animate']]
animate_tokens_df.to_csv(anim_tokens_df_format.format(lang, lang))

# After this step we did manual corrections and filtering.

# load the animate df back in
animate_tokens_df = pd.read_csv(anim_tokens_df_format.format(lang, lang))


# CREATE DISTRIBUTION SUMMARIES
def get_frequencies(df, log=True, animate_subset=False, combine_plurals=False):
    return_dict = {}

    total_count = df['count'].sum()
    if log: print('Total count: ', total_count)
    return_dict['Total count'] = total_count
    
    dist = {}
    numbers = POSSIBLE_NUMBERS
    genders = POSSIBLE_GENDERS 
    
    for number in numbers:
        for gender in genders:   
            if animate_subset and gender=='Neut': continue
            if number=='Plur' and lang=='nl': 
                count = df[df['number']==number]['count'].sum()
            else:
                
                count = df[
                (df['gender']==gender) & 
                (
                    df['number']==number
                )
                ]['count'].sum()
            freq = count / total_count
            if not gender: gender = ''            
            dist[gender + number] = freq
        
            if log:
                print('Count {}: {}'.format(gender + number, count))
                print('Freq {}: {}'.format(gender + number, freq))
            return_dict['Count {}'.format(gender + number)] = count
            return_dict['Freq {}'.format(gender + number)] = freq

    # Clean up if german
    if lang == 'de' and combine_plurals:
        return_dict['Count Plur'] = sum([return_dict['Count {}Plur'.format(gender)] for gender in genders])
        return_dict['Freq Plur'] = sum([return_dict['Freq {}Plur'.format(gender)] for gender in genders])

        for k in ['Count {}'.format(gender + 'Plur') for gender in genders] + ['Freq {}'.format(gender + 'Plur') for gender in genders]:
            return_dict.pop(k)

    distn = [val for val in list(return_dict.values()) if 1 > val > 0]
    h = entropy(distn, base=2)
    return_dict['Entropy'] = h
    
    # Print KL
    total_values = len(distn)
    kl = entropy(distn, [1/total_values for _ in range(total_values)], base=2)
    return_dict['KL'] = kl

    return_dict['dist'] = distn
    
    return return_dict


types_df = all_tokens_df.drop(['text'], axis=1)
types_df = types_df.groupby(['lemma', 'gender', 'number', 'animate']).agg('sum').reset_index()
types_df['count'] = 1

animate_types_df = animate_tokens_df.drop(['text'], axis=1)
animate_types_df = animate_types_df.groupby(['lemma', 'gender', 'number', 'animate']).agg('sum').reset_index()
animate_types_df['count'] = 1

token_dist = print_frequencies(all_tokens_df, combine_plurals=True)
type_dist = print_frequencies(types_df, combine_plurals=True)
anim_token_dist = get_frequencies(animate_tokens_df, animate_subset=True)
anim_type_dist = get_frequencies(animate_types_df, animate_subset=True)


# Put together and save
dist_summary = pd.DataFrame({'token': token_dist, 'type': type_dist, 'anim_token': anim_token_dist, 'anim_type': anim_type_dist}).T
dist_summary.to_csv(dist_summary_format.format(lang))