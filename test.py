import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import re
from unidecode import unidecode
from sklearn.pipeline import Pipeline

# --- Enhanced Data Loading ---
def load_data(path, file_type='csv'):
    """Load data with improved error handling and automatic encoding detection"""
    try:
        if file_type == 'csv':
            return pd.read_csv(
                path,
                sep=';',
                engine='python',
                on_bad_lines='warn',  # Better for debugging than 'skip'
                encoding='utf-8-sig',
                dtype=str
            )
        elif file_type == 'excel':
            return pd.read_excel(path, engine='openpyxl')
        raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return pd.DataFrame()

# Load data
df1 = load_data(r'C:\Users\carle\Desktop\python_work\p2\G5P2\Arbejdsordre med afskrevet reservedele.xlsx', file_type='excel')
df2 = load_data(r'C:\Users\carle\Desktop\python_work\p2\G5P2\Alle Arbejdsordre.xlsx', file_type='excel')

# --- Robust Data Cleaning ---
def clean_text(text):
    """Normalize text with improved cleaning"""
    if pd.isna(text): return ''
    text = unidecode(text).lower()  # Handle special characters
    text = re.sub(r'[^\w\s-]', '', text)  # Remove punctuation
    return text.strip()

# Clean column names
for df in [df1, df2]:
    df.columns = [clean_text(col) for col in df.columns]

# --- Enhanced Merging ---
# Check for common columns before merge
common_cols = list(set(df1.columns) & set(df2.columns))
print(f"Common columns for merging: {common_cols}")

merged_df = pd.merge(
    df2, 
    df1, 
    left_on='Work Order Number', 
    right_on='Work Order', 
    how='inner',
    validate='one_to_one'  # Ensure expected merge behavior
)

# --- Improved Categorization ---
CATEGORIES = {
    'Filtere': ['filter', 'filtrat'],
    'Ventiler': ['ventil', 'valve'],
    'Slanger': ['slange', 'hose'],
    'Motorer': ['motor', 'engine'],
    'Mælkemoduler': ['mælk', 'milk'],
    'Pumper': ['pumpe', 'pump']
}

def categorize_spare_part(name):
    """More maintainable categorization system"""
    name = clean_text(name)
    for category, keywords in CATEGORIES.items():
        if any(keyword in name for keyword in keywords):
            return category
    return 'Andet'

# --- Enhanced Feature Engineering ---
danish_stopwords = set(stopwords.words('danish')) | {'del', 'nummer', 'nr'}

# Create pipeline for text processing
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words=danish_stopwords,
        ngram_range=(1, 2),
        max_df=0.85,  # Ignore overly common terms
        min_df=2      # Ignore rare terms
    )),
    ('pca', PCA(n_components=0.95))  # Dimensionality reduction
])

# --- Optimal Cluster Determination ---
def find_optimal_clusters(data, max_k=10):
    """Use elbow method and silhouette score to find optimal clusters"""
    distortions = []
    silhouette_scores = []
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data)
        
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
    
    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(2, max_k+1), distortions, 'b-o')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(range(2, max_k+1), silhouette_scores, 'r-o')
    ax2.set_ylabel('Silhouette Score', color='r')
    
    plt.title('Elbow Method and Silhouette Score Analysis')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Data preparation
    filtered_df = merged_df.dropna(subset=['name', 'instructions']).copy()
    filtered_df['hovedkategori'] = filtered_df['name'].apply(categorize_spare_part)
    
    # Feature transformation
    X_transformed = text_pipeline.fit_transform(filtered_df['name'])
    
    # Cluster optimization
    find_optimal_clusters(X_transformed)
    
    # Final clustering with optimal k
    optimal_k = 6  # Set based on analysis
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    filtered_df['cluster'] = kmeans.fit_predict(X_transformed)
    
    # Add cluster centers for interpretation
    cluster_centers = text_pipeline.named_steps['pca'].inverse_transform(kmeans.cluster_centers_)
    terms = text_pipeline.named_steps['tfidf'].get_feature_names_out()
    
    print("\nTop terms per cluster:")
    for i, center in enumerate(cluster_centers):
        top_terms = [terms[ind] for ind in np.argsort(center)[-5:]]
        print(f"Cluster {i}: {', '.join(top_terms)}")
    
    # --- Enhanced Visualization ---
    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        x=X_transformed[:, 0], 
        y=X_transformed[:, 1],
        hue=filtered_df['cluster'],
        palette='viridis',
        legend='full'
    )
    plt.title('PCA Visualization of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Cluster distribution by category
    cross_tab = pd.crosstab(filtered_df['hovedkategori'], filtered_df['cluster'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
    plt.title('Cluster Distribution Across Categories')
    plt.xlabel('Cluster')
    plt.ylabel('Main Category')
    plt.show()

