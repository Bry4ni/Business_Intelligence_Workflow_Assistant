import pandas as pd
import random
from datetime import datetime, timedelta

def generate_sample_sales_data(filename='data/sample_sales.csv', num_records=200):
    products = ['Widget A', 'Widget B', 'Gadget X', 'Gadget Y']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    base_date = datetime.today()

    for _ in range(num_records):
        record = {
            'Date': (base_date - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d'),
            'Product': random.choice(products),
            'Region': random.choice(regions),
            'Units Sold': random.randint(1, 50),
            'Unit Price': round(random.uniform(10.0, 100.0), 2),
        }
        record['Revenue'] = round(record['Units Sold'] * record['Unit Price'], 2)
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Sample data written to {filename}")

# Run this file to generate sample data
if __name__ == "__main__":
    generate_sample_sales_data()
