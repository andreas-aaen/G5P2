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
desc_w_index_path = os.path.join(parent_dir, 'docs_to_label.json')
document_topic_data_path = os.path.join(parent_dir, 'document_topic_data.json')

load_merged_df_pickle_path = os.path.join(current_dir, 'merged_df_cached.pkl')

bw4_parts = load_data(bw4_parts_path, enc='utf-8-sig')
with open(desc_w_index_path, 'r') as f:
    desc_w_index = json.load(f)
with open(document_topic_data_path, 'r') as f:
    document_topic_data = json.load(f)
with open(load_merged_df_pickle_path, 'rb') as f:
        merged_df = pickle.load(f)

relevant_indices = {item[0] for item in desc_w_index}
index_map = {original_index: list_pos for list_pos, original_index in enumerate(relevant_indices)}
for original_index in relevant_indices:
    if original_index in index_map:
        list_pos = index_map[original_index]
        if list_pos < len(document_topic_data):
            replacement_modules = []
            try:
                lookup_val = merged_df.loc[original_index, 'Supplier Item number (Product) (Product)']
                if not pd.isna(lookup_val):
                    condition = (bw4_parts['Article no.'] == lookup_val)
                    module_types_series = bw4_parts.loc[condition, 'module_type']
                    if not module_types_series.empty:
                        unique_types = module_types_series.dropna().astype(str).unique()
                        replacement_modules = unique_types.tolist()
            except KeyError:
                print(f"Index {original_index} blev ikke fundet i merged_df")
            document_topic_data[list_pos]['replacement_part_modules'] = replacement_modules
        else:
            print(f"list_pos {list_pos} er ude af scope for document_topic_data (strørrelse {len(document_topic_data)}). Index: {original_index}")
with open(document_topic_data_path, 'w') as f:
    json.dump(document_topic_data, f, indent = 4)