import re
from rapidfuzz import process, fuzz
import pandas as pd

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
