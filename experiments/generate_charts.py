import sqlite3
import pandas as pd

def analyze():
    conn = sqlite3.connect("data/prompt_experiments.db")
    df_runs = pd.read_sql_query("SELECT * FROM rag_benchmark_runs WHERE condition IN ('baseline', 'rag')", conn)
    
    print("--- STEP COUNT STATISTICS ---")
    print(df_runs.groupby(['condition', 'task_id'])['steps'].mean())
    print(df_runs.groupby(['condition'])['steps'].mean())
    
    print("\n--- COST USD STATISTICS (as in DB) ---")
    print(df_runs.groupby(['condition', 'task_id'])['cost_usd'].mean())
    print(df_runs.groupby(['condition'])['cost_usd'].mean())
    
    print("\n--- OUTCOMES ---")
    print(df_runs.groupby(['condition', 'task_id', 'outcome']).size())
    
    conn.close()

if __name__ == "__main__":
    analyze()
