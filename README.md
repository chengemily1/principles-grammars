# Code and data for Principles of Semantic and Functional Efficiency in Grammatical Patterning

## Organization
- <code>scripts/</code> all python files for loading and processing the wikipedia data
- <code>distributions/</code> the processed distributions used to load the data
- <code>german_animate_lemmas_from_english.txt</code> the list of German animate lemmas, translated from English, due to poor coverage in German WordNet.

## Processing data
__Supported languages__: Spanish 'es', French 'fr', Italian 'it', Catalan 'ca', German 'de', English 'en', Swedish 'sv', Dutch 'nl', Slovene 'sl', Polish 'pl', Arabic 'ar', and Hebrew 'he'.
In modifying the python scripts, filepaths need to be changed.

- To save the data and convert Wikipedia to conllu, the entry point is <code>scripts/wiki_to_conllu.py</code>.
- Then, the nouns preprocessing is done in  <code>scripts/extract_noun_distributions.py</code>.
- To obtain the animate nouns, an extra preprocessing step was done to remedy incorrect gender tagging in the animate subset; nouns were filtered and gender tags corrected by hand for Slovene, Polish, Hebrew, Arabic, and German.
- A sample plotting script for the entropy curve is given in <code>scripts/entropy_plot.ipynb</code>.
