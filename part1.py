import json
import pandas as pd
from collections import deque, defaultdict
from datetime import timedelta


def preprocess_transactions_from_jsonl(file_path: str) -> pd.DataFrame:
    """Reads the JSONL file and flattens it into a DataFrame of transactions."""

    transaction_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip())

            # Create a quick lookup map for paymentMethodId -> issuer
            issuer_map = {
                pm['paymentMethodId']: pm['paymentMethodIssuer']
                for pm in data.get('paymentMethods', [])
            }
            
            fraudulent = data.get('fraudulent', None)
            for tx in data.get('transactions', []):
                issuer = issuer_map.get(tx['paymentMethodId'])

                # Only include transactions where we can find the issuer
                if issuer:
                    transaction_list.append({
                        'transactionId': tx['transactionId'],
                        'transactionTime': pd.to_datetime(tx['loggedAt']),
                        'paymentMethodIssuer': issuer,
                        'fraudulent': fraudulent
                    })
    
    return pd.DataFrame(transaction_list)


def calculate_velocity_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the 24-hour transaction velocity feature for each issuer."""
    
    # Sort all transactions by time O(nlogn)
    df_sorted = df.sort_values('transactionTime').reset_index(drop=True)
    
    results = [] # A list to store issuer_velocity_24h
    window = deque()  # A queue to hold transactions within the 24-hour window
    issuer_counts = defaultdict(int) # A counter for issuers within the window
    
    # Iterate through sorted transactions using a sliding time window (O(n))
    for _, row in df_sorted.iterrows():
        print(row)
        current_time = row['transactionTime']
        current_issuer = row['paymentMethodIssuer']
        
        # Remove old transactions from the left side of the window
        while window and window[0]['transactionTime'] <= current_time - timedelta(hours=24):
            old_transaction = window.popleft()
            issuer_counts[old_transaction['paymentMethodIssuer']] -= 1
        
        # The current count for the issuer is feature value
        velocity = issuer_counts.get(current_issuer, 0)
        results.append(velocity)
        
        # Add the current transaction to the right side of the window
        window.append(row)
        issuer_counts[current_issuer] += 1
        
    df_sorted['issuer_velocity_24h'] = results
    return df_sorted

if __name__ == "__main__":
    DATASET = 'customers_generated_100000_seed42.jsonl'
    
    print("Preprocessing data from JSONL file")
    transactions_df = preprocess_transactions_from_jsonl(DATASET)
    print(f"{len(transactions_df)} transactions found")
    
    print("Calculating velocity feature using a sliding window")
    final_df = calculate_velocity_feature(transactions_df)
    
    print("Done. Saving output as training_data_with_velocity.csv")
    final_df.to_csv('training_data_with_velocity.csv', index=False)
