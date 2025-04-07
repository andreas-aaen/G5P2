import os
from pathlib import Path
from collections import Counter
import json
import pandas as pd
import dacy
from gensim import corpora
from gensim.models import LdaModel

from path_plus_word_counts import bow_create_path

dir_path = os.path.dirname(os.path.abspath(__file__))
unprocessed_doc = os.path.join(dir_path, 'document_for_processing.txt')
data_path = Path(unprocessed_doc)
contents_data = data_path.read_text()

print(contents_data)

nlp_dansk = dacy.load('large')

desired_tags = ['NOUN', 'VERB']
nouns_verbs_only = []
doc = nlp_dansk(contents_data)
for token in doc:
    word_class = token.pos_
    if word_class in desired_tags:
        nouns_verbs_only.append(token)

for i, word in enumerate(nouns_verbs_only):
    nouns_verbs_only[i] = nouns_verbs_only[i].lemma_.lower()

contents_path = bow_create_path('post_term_limit_')
post_lim_bow = json.loads(contents_path.read_text())

i = 0
for word in nouns_verbs_only:
    if word in post_lim_bow:
        nouns_verbs_only[i] = word
        i += 1

dictionary = corpora.Dictionary.load('lda_model.id2word')
bow = dictionary.doc2bow(nouns_verbs_only)
print(bow)
lda_model = LdaModel.load('lda_model.model')
topic_distribution = lda_model[bow]

print(topic_distribution)