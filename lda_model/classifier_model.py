import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Ignore warnings about metrics for labels with no predicted samples
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- 1. Load Data ---
data_file = 'document_topic_data.json'
try:
    with open(data_file, 'r') as f:
        data = json.load(f) # contains topic_distribution and replacement_part_modules
except FileNotFoundError:
    print(f"Error: {data_file} not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode {data_file}.")
    exit()

# Filter out entries that might be missing the required keys
filtered_data = [
    item for item in data
    if 'topic_distribution' in item and 'replacement_part_modules' in item
]

if not filtered_data:
    print("Error: No valid data found in the JSON file.")
    exit()

# --- 2. Prepare Features (X) and Labels (Y) ---

# Features (X): The topic distribution vectors
X = np.array([item['topic_distribution'] for item in filtered_data]) #

# Labels (Y): The lists of replacement part modules
Y_raw = [item['replacement_part_modules'] for item in filtered_data] #

# Check if we have enough data
if len(X) == 0 or len(Y_raw) == 0:
    print("Error: Not enough data after filtering.")
    exit()

# --- 3. Encode Multi-Labels ---
# Use MultiLabelBinarizer to convert lists of labels into binary vectors
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y_raw)

# Print the identified classes (unique modules)
print(f"Identified Classes (Modules): {mlb.classes_}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features (topics): {X.shape[1]}")
print(f"Number of unique labels (modules): {Y.shape[1]}")

if Y.shape[1] == 0:
    print("\nError: No replacement part modules found in the data. Cannot train a classifier.")
    exit()

# --- 4. Split Data ---
# Split into training and testing sets (e.g., 80% train, 20% test)
try:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
except ValueError as e:
    print(f"\nError during train/test split: {e}")
    print("This might happen if the dataset is too small.")
    exit()


# --- 5. Choose and Train Model ---
# Use OneVsRestClassifier with RandomForest as the base estimator
# OneVsRest trains a separate classifier for each label
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model = OneVsRestClassifier(base_classifier)

print("\nTraining the model...")
model.fit(X_train, Y_train)
print("Training complete.")

# --- 6. Make Predictions ---
print("Making predictions on the test set...")
Y_pred = model.predict(X_test)
Y_pred_proba = model.predict_proba(X_test) # Get probabilities if needed

# --- 7. Evaluate Model ---
print("\n--- Evaluation Results ---")

# Calculate metrics
# Accuracy Score (Subset Accuracy): Exact match of label sets
subset_accuracy = accuracy_score(Y_test, Y_pred)
# Hamming Loss: Fraction of labels that are incorrectly predicted
hamming = hamming_loss(Y_test, Y_pred)
# F1 Score (Macro): Average F1 score per label (unweighted)
f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
# F1 Score (Micro): Aggregate contributions of all classes for F1
f1_micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
# Jaccard Score (Macro): Average Jaccard similarity coefficient per label
jaccard_macro = jaccard_score(Y_test, Y_pred, average='macro', zero_division=0)
# Jaccard Score (Micro): Aggregate Jaccard similarity coefficient
jaccard_micro = jaccard_score(Y_test, Y_pred, average='micro', zero_division=0)


print(f"Subset Accuracy (Exact Match Ratio): {subset_accuracy:.4f}")
print(f"Hamming Loss (Lower is better):      {hamming:.4f}")
print(f"F1 Score (Macro):                    {f1_macro:.4f}")
print(f"F1 Score (Micro):                    {f1_micro:.4f}")
print(f"Jaccard Score (Macro):               {jaccard_macro:.4f}")
print(f"Jaccard Score (Micro):               {jaccard_micro:.4f}")

# Detailed classification report (Precision, Recall, F1 per label)
print("\nClassification Report (Per Label):")
# Ensure target_names length matches the number of labels (Y.shape[1])
if len(mlb.classes_) == Y.shape[1]:
     print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))
else:
     print("Warning: Mismatch between MLB classes and Y shape. Skipping per-label report.")
     # Fallback report without names
     print(classification_report(Y_test, Y_pred, zero_division=0))


# --- 8. Example Prediction (Optional) ---
print("\n--- Example Prediction ---")
# Show prediction for the first test sample
example_index = 0
if len(X_test) > example_index:
    example_features = X_test[example_index]
    example_true_labels_bin = Y_test[example_index]
    example_pred_labels_bin = Y_pred[example_index]

    # Inverse transform to get label names
    example_true_labels = mlb.inverse_transform(np.array([example_true_labels_bin]))
    example_pred_labels = mlb.inverse_transform(np.array([example_pred_labels_bin]))

    print(f"Features (Topic Distribution): {np.round(example_features, 3)}")
    print(f"True Modules:     {example_true_labels[0] if example_true_labels else 'None'}")
    print(f"Predicted Modules: {example_pred_labels[0] if example_pred_labels else 'None'}")
else:
    print("Not enough samples in the test set for an example.")