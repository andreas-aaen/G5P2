import pandas as pd
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # Skift til hurtigere model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# Trin 1: Indlæs data
def load_data(path):
    return pd.read_csv(
        path,
        sep=';',
        engine='python',
        on_bad_lines='skip',
        encoding='latin1',  # Tilføj eksplicit tegnsæt
        dtype='unicode'     # Undgå automatisk typegæt
    )

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
file_path = os.path.join(script_dir, 'Arbejdsordre med afskrevet reservedele.csv')
file2_path = os.path.join(script_dir, 'Alle Arbejdsordre.csv')

df1 = load_data(file_path)
df2 = load_data(file2_path)

# Trin 2: Dataforberedelse
# Fjern duplikerede reset_index kald
merged_df = pd.merge(df2, df1, left_on='Work Order Number', right_on='Work Order', how='inner')

# Fjern kun rækker hvor både Name og Instructions mangler én gang
filtered_df = merged_df.dropna(subset=['Name', 'Instructions']).copy()

# Trin 3: Analyse af data
print(f"Antal datapunkter: {len(filtered_df)}")
print(f"Antal unikke reservedele: {filtered_df['Name'].nunique()}")

# Filtrér væk sjældne reservedele (juster threshold efter behov)
name_counts = filtered_df['Name'].value_counts()
filtered_df = filtered_df[filtered_df['Name'].isin(name_counts[name_counts > 200].index)]

danish_stopwords = (stopwords.words('danish'))

# Trin 4: Tekstbehandling
vectorizer = TfidfVectorizer(
    max_features=5000,       # Reducer yderligere antallet af features
    stop_words=danish_stopwords, # Brug danske stopord hvis tilgængelige
    ngram_range=(1, 2)      # Tag bigrams med
)
X = vectorizer.fit_transform(filtered_df['Instructions'])

# Konverter til pandas DataFrame for hukommelsesoptimering
X_df = pd.DataFrame.sparse.from_spmatrix(X, columns=vectorizer.get_feature_names_out())

# Trin 5: Modelvalg
model = SGDClassifier(
    loss='log_loss',         
    max_iter=1000,
    n_jobs=-1,              
    early_stopping=True      
)

# Trin 6: Træning med progress monitoring 
X_train, X_test, y_train, y_test = train_test_split(
    X_df, 
    filtered_df['Name'],    # Brug direkte kategorier i stedet for LabelEncoder
    test_size=0.3,
    stratify=filtered_df['Name'],  # Vigtigt for ubalanceret data
    random_state=42
)

model.fit(X_train, y_train)

# Evaluer
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Generer en confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Visualiser confusion matrix som et heatmap
plt.figure(figsize=(10, 8))  # Justér størrelsen af heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Heatmap')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
    

