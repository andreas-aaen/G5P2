import os
from collections import Counter
import pandas as pd
import dacy
print("DaCy installed successfully!")
from gensim import models

# Skal betragtes som immutable (constants)(hyperparametre)
NUM_TOPICS = 20
TERM_APPEARANCES = 10
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
nlp_dansk = dacy.load("large")

# Trin 2: Dataforberedelse
# Fjern duplikerede reset_index kald
merged_df = pd.merge(df2, df1, left_on='Work Order Number', right_on='Work Order', how='inner')

# Fjern kun rækker hvor både Name og Instructions mangler én gang
filtered_df = merged_df.dropna(subset=['Name', 'Instructions']).copy()
# Laver en kopi af de første 100 rækker i filtered_df
# Bruges til at teste i en lille chunk,
# - da tid gør koden utestbar ved at apply dacy-Large til hele datasættet
filtered_df_copy = filtered_df.loc[:CHUNK_SIZE, ]

# Bruger dacy-Large til at filtrere dokumenterne ned til en liste af lister
# Hver nestede liste består kun af navne- og udsagnsord
desired_tags = ['NOUN', 'VERB']
nouns_verbs_only = []
for document in filtered_df_copy['Instructions (Work Order) (Work Order)']:
    doc = nlp_dansk(document)
    temp = []
    for token in doc:
        word_class = token.pos_
        if word_class in desired_tags:
            temp.append(token)
    if temp:
        nouns_verbs_only.append(temp)

# Bruger dacy-Large til at lemmatize hvert ord i de nestede lister
for i, bag_of_words in enumerate(nouns_verbs_only):
    for j, word in enumerate(bag_of_words):
        nouns_verbs_only[i][j] = nouns_verbs_only[i][j].lemma_

count_nested_list_elements(nouns_verbs_only)

# Gemmer en dictionary med word som key og optrædener som value
word_counts = Counter()
for bag_of_words in nouns_verbs_only:
    for word in bag_of_words:
        word_counts[word] += 1

# Tildeler hver bag-of-words en ny bag-of-word baseret på den originale,
# - hvor kun termer der optræder mere end TERM_APPEARANCES i alle dokumenter bliver gemt.
for i, bag_of_words in enumerate(nouns_verbs_only):
    for word in enumerate(bag_of_words):
        nouns_verbs_only[i] = [word for word in bag_of_words 
                               if word_counts[word] >= TERM_APPEARANCES]

# Brugt til at se hvor mange termer den ovenstående optimering gav 
count_nested_list_elements(nouns_verbs_only)

# Laver en mapping hvor hvert unikke term er key for en value som er et unikt integer-index.
dictionary = {i: word for i, word in enumerate(word_counts.keys())}
word_index = {v:k for k, v in dictionary.items()}

# Tæller hvor mange gange hvert term optræder i én bag-of-words (én række i data)
# Laver en ny bag-of-words for hver bag-of-words, hvor den nye bag-of-words
# - er den gamle, men hvert term er sat til termets encoding istedet,
# - og for hvert term indeholder bag-of-words nu en tuple bestående af (encoding, count),
# - hvor count er antal gange det ord optræder i det tilsvarende dokument (række i Bentax data) 
for i, bag_of_words in enumerate(nouns_verbs_only):
    counts = Counter(bag_of_words)
    nouns_verbs_only[i] = [(word_index[word], count) for word, count in counts.items()]

# Træner modellen (Lda), hvor føste argument er listen af lister, bestående af tuples (encoding, count)-format
# Andet argument er måden hvorpå modellen kan decode encodingen tilbage til ord-repræsentationen
# Sidste argument er antallet af topics som modellen skal modellere baseret på data fra føste argument
model = models.LdaModel(nouns_verbs_only, id2word=dictionary, num_topics=NUM_TOPICS)

# Printer output af modellen: En liste af tupler, hvor hver tuple er ét emne
# Hver tuple indeholder navne- og udsagnsord som er vigtige for det givne emne
# Skalaren er vigtigheden (for det givne emne) af det efterfølgende term
print(model.print_topics())