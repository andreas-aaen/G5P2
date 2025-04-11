import os
from pathlib import Path
import json
from collections import Counter

def bow_create_path(prefix):
    script_folder = os.path.dirname(os.path.abspath(__file__))
    bow_path = os.path.join(script_folder, prefix + 'bag_of_words.json')
    path = Path(bow_path)
    return path

def save_bow(bow, prefix):
    path = bow_create_path(prefix)
    contents = json.dumps(bow)
    path.write_text(contents)

def load_bow(version_prefix):
    path = bow_create_path(version_prefix)
    contents = path.read_text()
    bow = json.loads(contents)
    return bow

def create_word_counts():
    pre_bow = load_bow('pre_term_limit_')
    word_counts = Counter()
    for bag_of_words in pre_bow:
        for word in bag_of_words:
            word_counts[word] += 1
    return word_counts