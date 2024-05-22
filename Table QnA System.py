from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
import os
import sqlite3

# Load Tapex tokenizer and model
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

# Function to load dataset from CSV file
def load_csv_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except Exception as e:
        print("Error loading CSV dataset:", e)
        return None

# Function to load dataset from Excel file
def load_excel_dataset(file_path):
    try:
        dataset = pd.read_excel(file_path)
        return dataset
    except Exception as e:
        print("Error loading Excel dataset:", e)
        return None

# Function to load dataset from SQL database
def load_sql_dataset(database_path, table_name):
    try:
        conn = sqlite3.connect(database_path)
        query = f"SELECT * FROM {table_name}"
        dataset = pd.read_sql_query(query, conn)
        return dataset
    except Exception as e:
        print("Error loading SQL dataset:", e)
        return None

# Function to generate SQL query from natural language query
def generate_sql(query, table, column_names, max_length=512):
    encoding = tokenizer(table=table, query=query, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(**encoding)
    sql_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return sql_query

def main():
    # Prompt user to enter dataset path
    dataset_path = input("Enter dataset path: ")
    
    # Check if the provided path is a directory or file
    if os.path.isdir(dataset_path):
        # Load and process datasets from the directory
        datasets = []
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            if file_name.endswith(".csv"):
                dataset = load_csv_dataset(file_path)
            elif file_name.endswith(".xlsx"):
                dataset = load_excel_dataset(file_path)
            elif file_name.endswith(".db"):
                database_name = os.path.splitext(file_name)[0]
                table_name = input(f"Enter table name in {file_name}: ")
                dataset = load_sql_dataset(file_path, table_name)
            else:
                print(f"Unsupported file format: {file_name}")
                continue
            
            if dataset is not None:
                datasets.append((file_name, dataset))
        
        # Display loaded datasets
        print("Loaded datasets:")
        for dataset_name, dataset in datasets:
            print(f"- {dataset_name}: {len(dataset)} rows")
    elif os.path.isfile(dataset_path):
        # Load single dataset file
        file_name = os.path.basename(dataset_path)
        if file_name.endswith(".csv"):
            dataset = load_csv_dataset(dataset_path)
        elif file_name.endswith(".xlsx"):
            dataset = load_excel_dataset(dataset_path)
        elif file_name.endswith(".db"):
            table_name = input(f"Enter table name in {file_name}: ")
            dataset = load_sql_dataset(dataset_path, table_name)
        else:
            print(f"Unsupported file format: {file_name}")
            return
        
        if dataset is not None:
            datasets = [(file_name, dataset)]
            print(f"Loaded dataset: {file_name}: {len(dataset)} rows")
    else:
        print("Invalid dataset path. Exiting.")
        return

    # Prompt user to enter natural language queries
    while True:
        query = input("Enter a natural language query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Prompt user to choose a dataset
        if len(datasets) > 1:
            print("Available datasets:")
            for i, (dataset_name, _) in enumerate(datasets):
                print(f"{i + 1}. {dataset_name}")
            dataset_choice = input("Choose a dataset (enter the number): ")
            
            # Validate dataset choice
            try:
                dataset_index = int(dataset_choice) - 1
                dataset_name, dataset = datasets[dataset_index]
            except (ValueError, IndexError):
                print("Invalid dataset choice. Please choose a valid number.")
                continue
        else:
            dataset_name, dataset = datasets[0]
        
        # Generate SQL query
        column_names = dataset.columns.tolist()
        sql_query = generate_sql(query, dataset, column_names)
        print("Answer of Query:", sql_query)

# Example usage
if __name__ == "__main__":
    main()
