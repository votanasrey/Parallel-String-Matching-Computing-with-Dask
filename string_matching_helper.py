import re
from rapidfuzz import process, fuzz
import pandas as pd
from string_grouper import match_strings, match_most_similar, group_similar_strings, compute_pairwise_similarities, StringGrouper
import concurrent.futures


def clean_text(text):
    
    if not isinstance(text, str):
        text = ''
    cleaned_text = re.sub(r'[^a-zA-Z.\- ]+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def apply_rapidfuzz_matching(df, aba_cust_list):
    df['bakong_cust'] = df['bakong_cust'].astype(str)

    def apply_fuzzy(x):
        match = process.extractOne(x, aba_cust_list, scorer=fuzz.token_set_ratio, score_cutoff=95)
        return match if match else (None, None, None)  
    
    df['match_results'] = df['bakong_cust'].apply(apply_fuzzy)
    
    results = pd.DataFrame(df['match_results'].tolist(), index=df.index)
    
    results.columns = ['aba_cust', 'score', 'other']

    final_results = df[['bakong_cust']].join(results[['aba_cust', 'score']])
    
    # Drop rows without a match
    return final_results.dropna(subset=['aba_cust', 'score'])

def chunk_data_to_csv(file_path, chunk_size, output_dir):
    
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    for i, chunk in enumerate(chunk_iterator):
        output_file = f"{output_dir}/chunk_{i}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Chunk {i} saved as {output_file}")

def string_matching(series1, series2):
    series1 = series1.astype(str)
    series2 = series2.astype(str)

    results = []
    for i in series1:
        for j in series2:
            result = compute_pairwise_similarities(pd.Series([i]), pd.Series([j]))
            results.extend(result)

    return pd.DataFrame(results)