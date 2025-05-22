import os
import pandas as pd
import json
import pickle

def load_data(path, enc='latin1'):
    return pd.read_csv(
        path,
        sep=';',
        engine='python',
        on_bad_lines='skip',
        encoding=enc,
        dtype='unicode'
    )

current_filepath = os.path.abspath(__file__)
current_dir = os.path.dirname(current_filepath)
parent_dir = os.path.dirname(current_dir)

bw4_parts_path = os.path.join(parent_dir, 'reservedelslister', 'bw4', 'bw4.csv')
bw4c_parts_path = os.path.join(parent_dir, 'reservedelslister', 'BW4c', 'BW4c.csv')
desc_w_index_path = os.path.join(parent_dir, 'docs_to_label.json')
document_topic_data_path = os.path.join(parent_dir, 'document_topic_data.json')

load_merged_df_pickle_path = os.path.join(current_dir, 'merged_df_cached.pkl')

try:
    bw4_parts = load_data(bw4_parts_path, enc='utf-8-sig')
    print(f"BW4-liste loaded fra: {bw4_parts_path}")
except FileNotFoundError:
    print(f"Fejl: kunne ikke finde filen på lokation; {bw4_parts_path}")
    exit()
try:
    bw4c_parts = load_data(bw4c_parts_path, enc='utf-8-sig')
    print(f"BW4c-liste loaded fra: {bw4c_parts_path}")
except FileNotFoundError:
    print(f"Fejl: kunne ikke finde filen på lokation; {bw4_parts_path}")
    exit()

try:
    with open(desc_w_index_path, 'r') as f:
        desc_w_index = json.load(f)
    with open(document_topic_data_path, 'r') as f:
        original_document_topic_data = json.load(f)
    with open(load_merged_df_pickle_path, 'rb') as f:
            merged_df = pickle.load(f)
except FileNotFoundError as e:
    print(f"Fejl: kunne ikke finde filen på lokation; {e}")
    exit()
except json.JSONDecodeError as e:
    print(f"Fejl: kunne ikke decode JSON-filen: {e}")
    exit()
except pickle.PickleError as e:
    print(f"Fejl: kunne ikke loade pickle-filen: {e}")
    exit()

# Ny liste der kun skal gemme dictionaries for elementer hvor der blev fundet modul-kategorier i bw4 og/eller bw4c datasættet
altered_document_topic_data = []

for list_pos, item_in_desc_w_index in enumerate(desc_w_index):
    original_index_for_merged_df = item_in_desc_w_index[0]

    if list_pos < len(original_document_topic_data):
        current_doc_data = original_document_topic_data[list_pos].copy()
        replacement_modules_set = set()

        try:
            # Use original_index_for_merged_df to lookup in merged_df
            lookup_val = merged_df.loc[original_index_for_merged_df, 'Supplier Item number (Product) (Product)']

            if not pd.isna(lookup_val):
                def get_modules_from_parts_df(parts_df, article_no_val):
                    modules = set()
                    if 'Article no.' not in parts_df.columns or 'module_type' not in parts_df.columns:
                        print(f"Fejl: df mangler enten 'Article no.' eller 'module_type' kolonne.")
                        return modules
                    
                    condition = (parts_df['Article no.'] == article_no_val)
                    module_types_series = parts_df.loc[condition, 'module_type']
                    if not module_types_series.empty:
                        unique_types = module_types_series.dropna().astype(str).unique()
                        for type_module in unique_types:
                            modules.add(type_module)
                    return modules
                    
                modules_from_bw4 = get_modules_from_parts_df(bw4_parts, lookup_val)
                replacement_modules_set.update(modules_from_bw4)

                modules_from_bw4c = get_modules_from_parts_df(bw4c_parts, lookup_val)
                replacement_modules_set.update(modules_from_bw4c)

        except KeyError:
            print(f"Index {original_index_for_merged_df} (for document_topic_data entry {list_pos}) was not found in merged_df.")
        except Exception as e:
            print(f"An error occurred for document {list_pos} with original index {original_index_for_merged_df}: {e}")

        replacement_modules = list(replacement_modules_set)

        # Tilføjer kun til den nye liste hvis der er elementer i replacement_modules listen
        if replacement_modules:
            if 'topic_distribution' not in current_doc_data:
                print(f"Fejl: topic_distribution key mangler i original_document_topic_data entry {list_pos}. Fejl i træning fra lda.")
                current_doc_data['topic_distribution'] = []

            current_doc_data['replacement_part_modules'] = replacement_modules

        else:
            print(f"Info: Ingen replacement modules kunne findes for document_topic_data entry {list_pos} (originalt index i merged_df: {original_index_for_merged_df}). Entry vil ikke blive inkluderet i træning.")
        
    else:
        print(f"Fejl: Index {list_pos} fra desc_w_index er højere end index-rækken for original_document_topic_data (af længde: {len(original_document_topic_data)}). Entry fra desc_w_index bliver ikke inkluderet i træning.")

print(f"\nOriginalt antal entries for document_topic_data: {len(original_document_topic_data)}")
print(f"Opdateret antal entries med modul-kategorier fundet: {len(altered_document_topic_data)}")

# Save the updated document_topic_data
with open(document_topic_data_path, 'w') as f:
    json.dump(altered_document_topic_data, f, indent=4)

print(f"\ndocument_topic_data.json opdateret og gemt med {len(altered_document_topic_data)} entries.")