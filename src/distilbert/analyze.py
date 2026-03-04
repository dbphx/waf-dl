import pandas as pd
import json

def analyze_dataset(file_path, is_attack):
    print(f"--- Analyzing {file_path} ---")
    df = pd.read_csv(file_path, sep=';', dtype=str, on_bad_lines='skip')
    print(f"Total Rows: {len(df)}")
    print(df.head())
    
    if 'rule_names' in df.columns:
        print("\nTop 10 Rule Names:")
        print(df['rule_names'].value_counts().head(10))
    else:
        print(f"\nNo 'rule_names' column found. Columns: {df.columns}")
        
    # See what typical sequence looks like
    for idx, row in df.head(5).iterrows():
        method = row.get('http_method', '')
        path = row.get('http_path', '')
        query = row.get('http_query', '')
        print(f"Sample {idx}: {method} {path} {query}")

if __name__ == "__main__":
    analyze_dataset('../../data/normal.csv', False)
    print("\n\n")
    analyze_dataset('../../data/attack.csv', True)
