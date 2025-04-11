import os
from pathlib import Path
from path_plus_word_counts import bow_create_path, save_bow, load_bow, create_word_counts

import json
from collections import Counter
from gensim.models import LdaModel
from gensim import corpora
from gensim.models import TfidfModel
from path_plus_word_counts import create_word_counts, bow_create_path

# Skal betragtes som immutable (constants)(hyperparametre)
NUM_TOPICS = 20

contents_path = bow_create_path('post_term_limit_')
nouns_verbs_only = json.loads(contents_path.read_text())

unique_set = set()
for bag_of_words in nouns_verbs_only:
    for word in bag_of_words:
        unique_set.add(word)

print(len(unique_set))

# Laver en mapping hvor hvert unikke term er key for en value som er et unikt integer-index.
'''dictionary = {i: word for i, word in enumerate(create_word_counts().keys())}
word_index = {v:k for k, v in dictionary.items()}'''
# ny ækvivalent
gensim_dictionary = corpora.Dictionary(nouns_verbs_only)
word_index = gensim_dictionary.token2id

# Tæller hvor mange gange hvert term optræder i én bag-of-words (én række i data)
# Laver en ny bag-of-words for hver bag-of-words, hvor den nye bag-of-words
# - er den gamle, men hvert term er sat til termets encoding istedet,
# - og for hvert term indeholder bag-of-words nu en tuple bestående af (encoding, count),
# - hvor count er antal gange det ord optræder i det tilsvarende dokument (række i Bentax data) 
for i, bag_of_words in enumerate(nouns_verbs_only):
    counts = Counter(bag_of_words)
    nouns_verbs_only[i] = [(word_index[word], count) for word, count in counts.items()]

tfidf = TfidfModel(nouns_verbs_only)
corpus_tfidf = tfidf[nouns_verbs_only]

# Træner modellen (Lda), hvor føste argument er listen af lister, bestående af tuples (encoding, count)-format
# Andet argument er måden hvorpå modellen kan decode encodingen tilbage til ord-repræsentationen
# Sidste argument er antallet af topics som modellen skal modellere baseret på data fra føste argument
#model = models.LdaModel(nouns_verbs_only, id2word=dictionary, num_topics=NUM_TOPICS)
lda_model = LdaModel(corpus_tfidf, id2word=gensim_dictionary, num_topics=NUM_TOPICS)
lda_model.save('lda_model.model')
gensim_dictionary.save('lda_model.id2word')

# Printer output af modellen: En liste af tupler, hvor hver tuple er ét emne
# Hver tuple indeholder navne- og udsagnsord som er vigtige for det givne emne
# Skalaren er vigtigheden (for det givne emne) af det efterfølgende term
print(lda_model.print_topics())

# Printer distributionen af topics over det første dokument
print(f'\n\n{lda_model.get_document_topics(corpus_tfidf[0], minimum_probability=0.0)}')

data_for_training = []
for doc in corpus_tfidf:
    topic_distribution = lda_model.get_document_topics(doc, minimum_probability=0.0)

    topic_vec = [0] * lda_model.num_topics
    for topic_id, probability in topic_distribution:
        topic_vec[topic_id] = float(probability)

    data_for_training.append({'topic_distribution': topic_vec})

with open('document_topic_data.json', 'w') as file:
    json.dump(data_for_training, file, indent=4)