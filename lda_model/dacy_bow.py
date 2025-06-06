import os
from pathlib import Path
import json
import pickle
from collections import Counter
import re
import pandas as pd
import dacy

from path_plus_word_counts import bow_create_path, save_bow, load_bow, create_word_counts

# Skal betragtes som immutable (constants)(hyperparametre)
TERM_APPEARANCES = 2
CHUNK_SIZE = 22_000

# Trin 1: Indlæs data
def load_data(path, enc='latin1'):
    return pd.read_csv(
        path,
        sep=';',
        engine='python',
        on_bad_lines='skip',
        encoding=enc,
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
bw4_parts_path = os.path.join(script_dir, 'reservedelslister/bw4/bw4.csv')

df1 = load_data(file_path)
df2 = load_data(file2_path)
bw4_parts = load_data(bw4_parts_path, enc = 'utf-8-sig')
merged_df_pickle_path = os.path.join(script_dir, "augment_document_topic_data/merged_df_cached.pkl")

#Indlæser model "large" af dacy-library i hukommelsen
nlp_dansk = dacy.load('large', disable=['ner'])

# Trin 2: Dataforberedelse
# Fjern duplikerede, reset_index kald
merged_df = pd.merge(df2, df1, left_on='Work Order Number', right_on='Work Order', how='inner')
print(f"Antal rækker: {len(merged_df['Supplier Item number (Product) (Product)'])}") #Antal

# Fjern trailing 'R', så vi kan lave lookup i vores parts-lists dokumenter
print(f"værdier før:")
print(merged_df['Supplier Item number (Product) (Product)'].head(20))
merged_df['Supplier Item number (Product) (Product)'] = merged_df['Supplier Item number (Product) (Product)'].apply(lambda x: re.sub(r'R$', '', x) if pd.notna(x) else x)
print(f"Antal rækker: {len(merged_df['Supplier Item number (Product) (Product)'])}") #Antal
print(f"værdier efter:")
print(merged_df['Supplier Item number (Product) (Product)'].head(20))

# Filtrerer df så kun rækker hvor formattet på item number matcher formattet i udleveret datablad
print(f"\nAntal rækker før filtrering på supplier item number: {len(merged_df)}")
condition_not_na = merged_df['Supplier Item number (Product) (Product)'].notna()
condition_is_digit = merged_df['Supplier Item number (Product) (Product)'].str.isdigit()
condition_is_len_6 = merged_df['Supplier Item number (Product) (Product)'].str.len() == 6
row_to_keep = condition_not_na & condition_is_digit & condition_is_len_6
merged_df = merged_df[row_to_keep].copy()
print(f"\nAntal rækker efter filtrering på supplier item number: {len(merged_df)}")
print(f"Antal rækker: {len(merged_df['Supplier Item number (Product) (Product)'])}") #Antal

# Gemmer en pickle-fil af den endelige version af merged_df 
with open(merged_df_pickle_path, 'wb') as f:
        pickle.dump(merged_df, f)

# Fjern kun rækker hvor enten asset category eller instructions mangler
filtered_df = merged_df.dropna(subset=['Primær Asset Kategori (Work Order) (Work Order)', 'Instructions (Work Order) (Work Order)']).copy()
print(f"Antal rækker: {len(filtered_df['Supplier Item number (Product) (Product)'])}") #Antal

# Fjern rækker der ikke indeholder 'BW3' eller 'BW4' i primær asset category
thermoplan_filter = filtered_df['Primær Asset Kategori (Work Order) (Work Order)'].str.contains('BW4', na=False)
filtered_df = filtered_df[thermoplan_filter].copy()
print(len(filtered_df))
print(f"Antal rækker: {len(filtered_df['Supplier Item number (Product) (Product)'])}") #Antal

# Fjerner alle undtagen første instans af duplikerede fejlbeskrivelser fra listen over dokumenter
filtered_df_copy = filtered_df.drop_duplicates(subset=['Instructions (Work Order) (Work Order)'])
print(f"Antal rækker: {len(filtered_df_copy['Supplier Item number (Product) (Product)'])}") #Antal
filtered_df_copy = filtered_df_copy.loc[ : , ]

# En liste af instruktionerne der bliver brugt til bow's før teksten bliver transformeret
pre_trans_instructions = filtered_df_copy['Instructions (Work Order) (Work Order)'].tolist()

# Bruger clean_whitespace funtionen på et element i filtered_df_copy ad gangen
filtered_df_copy['Instructions (Work Order) (Work Order)'] = filtered_df_copy['Instructions (Work Order) (Work Order)'].apply(clean_whitespace)

# Bruger dacy-Large til at filtrere dokumenterne ned til en liste af lister
# Hver nestede liste (BOW) består kun af navne- og udsagnsord
desired_tags = ['NOUN', 'VERB']
texts = filtered_df_copy['Instructions (Work Order) (Work Order)'].tolist()
print(len(texts)) #Længde
nouns_verbs_only = [[token.lemma_.lower() for token in nlp_dansk(doc) if token.pos_ in desired_tags] for doc in texts]
print(len(nouns_verbs_only)) #Længde
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
desc_w_index = []
if len(texts) != len(filtered_df_copy):
    print("Fejl: længden på 'texts' matcher ikke længden på 'filtered_df_copy'")
    print(f"Længden på texts: {len(texts)}\nLængden på filtered_df_copy: {len(filtered_df_copy)}")
for i in range(len(filtered_df_copy)):
    actual_df_index_label_numpy = filtered_df_copy.index[i]
    actual_df_index_label_pyint = int(actual_df_index_label_numpy)
    text_content_for_json = texts[i]
    desc_w_index.append((actual_df_index_label_pyint, text_content_for_json))

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
        nouns_verbs_only[i] = [word for word in bag_of_words if word_counts[word] >= TERM_APPEARANCES]

# Brugt til at se hvor mange termer den ovenstående optimering gav 
count_nested_list_elements(nouns_verbs_only)

save_bow(nouns_verbs_only, 'post_term_limit_')