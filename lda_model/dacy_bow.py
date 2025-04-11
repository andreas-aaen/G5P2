import os
from pathlib import Path
import json
from collections import Counter
import re
import pandas as pd
import dacy

from path_plus_word_counts import bow_create_path, save_bow, load_bow, create_word_counts

# Skal betragtes som immutable (constants)(hyperparametre)
TERM_APPEARANCES = 2
CHUNK_SIZE = 100

# Trin 1: Indlæs data
def load_data(path):
    return pd.read_csv(
        path,
        sep=';',
        engine='python',
        on_bad_lines='skip',
        encoding='latin1',
        dtype='unicode'
    )

# Fjerner alle whitespace-karakterer samt leading og trailing whitespace
def clean_whitespace(texts):
    texts = re.sub(r'\s+', ' ', texts).strip()
    return texts

# Tæller samlet antal elementer i de nestede lister
# Kan kaldes for at se hvor mange termer optimeringer har elimineret
# Optimeringer kan bl.a. være stemming eller at filtrere baseret på TERM_APPEARANCES
def count_nested_list_elements(list_of_lists):
    count = 0
    for bag_of_words in list_of_lists:
        for word in bag_of_words:
            count += 1
    print(count)

#Boiler-plate for at sætte path object til data-filerne, og give dem et variabel-navn
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Arbejdsordre med afskrevet reservedele.csv')
file2_path = os.path.join(script_dir, 'Alle Arbejdsordre.csv')

df1 = load_data(file_path)
df2 = load_data(file2_path)

#Indlæser model "large" af dacy-library i hukommelsen
nlp_dansk = dacy.load('large', disable=['ner'])

# Trin 2: Dataforberedelse
# Fjern duplikerede reset_index kald
merged_df = pd.merge(df2, df1, left_on='Work Order Number', right_on='Work Order', how='inner')

# Fjern kun rækker hvor både Name og Instructions mangler én gang
filtered_df = merged_df.dropna(subset=['Name', 'Instructions']).copy()

# Fjerner alle undtagen første instans af duplikerede fejlbeskrivelser fra listen over dokumenter
filtered_df_copy = filtered_df.drop_duplicates(subset=['Instructions (Work Order) (Work Order)'])
filtered_df_copy = filtered_df_copy.loc[:CHUNK_SIZE, ]

# En liste af instruktionerne der bliver brugt til bow's før teksten bliver transformeret
pre_trans_instructions = filtered_df_copy['Instructions (Work Order) (Work Order)'].tolist()

# Bruger clean_whitespace funtionen på et element i filtered_df_copy ad gangen
filtered_df_copy['Instructions (Work Order) (Work Order)'] = filtered_df_copy['Instructions (Work Order) (Work Order)'].apply(clean_whitespace)

# Bruger dacy-Large til at filtrere dokumenterne ned til en liste af lister
# Hver nestede liste består kun af navne- og udsagnsord
desired_tags = ['NOUN', 'VERB']
texts = filtered_df_copy['Instructions (Work Order) (Work Order)'].tolist()
nouns_verbs_only = [[token.lemma_.lower() for token in nlp_dansk(doc) if token.pos_ in desired_tags] for doc in texts]
'''for document in filtered_df_copy['Instructions (Work Order) (Work Order)']:
    doc = nlp_dansk(document)
    temp = []
    for token in doc:
        word_class = token.pos_
        if word_class in desired_tags:
            temp.append(token)
    if temp:
        nouns_verbs_only.append(temp)'''

# Mapper beskrivelser fra data der bruges til bow's til deres korresponderende første index i det originale df.
index_in_mdf = []
for instruction in pre_trans_instructions:
    index_in_mdf.append(merged_df['Instructions (Work Order) (Work Order)'].tolist().index(instruction))
desc_w_index = [(index_in_mdf[index], item) for index, item in enumerate(texts)]

# Dumper beskrivelser (samt index til originale df) fra alle de dokumenter hvorfra der bliver dannet bow's til en liste i en json-fil.
with open('docs_to_label.json', 'w') as file:
    json.dump(desc_w_index, file, indent=4)

# Bruger dacy-Large til at lemmatize hvert ord i de nestede lister og konvertere string til lowercase
'''for i, bag_of_words in enumerate(nouns_verbs_only):
    for j, word in enumerate(bag_of_words):
        nouns_verbs_only[i][j] = nouns_verbs_only[i][j].lemma_.lower()'''

save_bow(nouns_verbs_only, 'pre_term_limit_')

count_nested_list_elements(nouns_verbs_only)

# Gemmer en dictionary med term som key og optrædener som value
word_counts = create_word_counts()

# Tildeler hver bag-of-words en ny bag-of-word baseret på den originale,
# - hvor kun termer der optræder mere end TERM_APPEARANCES i alle dokumenter bliver gemt.
for i, bag_of_words in enumerate(nouns_verbs_only):
    for word in enumerate(bag_of_words):
        nouns_verbs_only[i] = [word for word in bag_of_words 
            if word_counts[word] >= TERM_APPEARANCES]

# Brugt til at se hvor mange termer den ovenstående optimering gav 
count_nested_list_elements(nouns_verbs_only)

save_bow(nouns_verbs_only, 'post_term_limit_')