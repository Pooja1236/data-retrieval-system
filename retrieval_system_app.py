import streamlit as st
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
import sqlite3
import os

# Load Tapex tokenizer and model
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

# Function to load dataset from CSV file
def load_csv_dataset(file):
    try:
        dataset = pd.read_csv(file)
        return dataset
    except Exception as e:
        st.error(f"Error loading CSV dataset: {e}")
        return None

# Function to load dataset from Excel file
def load_excel_dataset(file):
    try:
        dataset = pd.read_excel(file)
        return dataset
    except Exception as e:
        st.error(f"Error loading Excel dataset: {e}")
        return None

# Function to load dataset from SQL database
def load_sql_dataset(file, table_name):
    try:
        conn = sqlite3.connect(file)
        query = f"SELECT * FROM {table_name}"
        dataset = pd.read_sql_query(query, conn)
        return dataset
    except Exception as e:
        st.error(f"Error loading SQL dataset: {e}")
        return None

# Function to generate SQL query from natural language query
def generate_sql(query, table, column_names, max_length=512):
    encoding = tokenizer(table=table, query=query, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(**encoding)
    sql_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return sql_query

# Streamlit App
st.title("Natural Language Query to SQL")

# Step 1: Upload Dataset
st.header("Step 1: Upload your dataset")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "db"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    if file_extension == '.csv':
        dataset = load_csv_dataset(uploaded_file)
    elif file_extension == '.xlsx':
        dataset = load_excel_dataset(uploaded_file)
    elif file_extension == '.db':
        table_name = st.text_input("Enter the table name in the database")
        if table_name:
            dataset = load_sql_dataset(uploaded_file, table_name)
    else:
        st.error("Unsupported file format")
    
    if dataset is not None:
        st.write("Dataset Loaded Successfully!")
        st.dataframe(dataset.head())

        # Step 2: Enter Natural Language Query
        st.header("Step 2: Enter your query")
        query = st.text_input("Enter a natural language query")

        if query:
            # Generate SQL query
            column_names = dataset.columns.tolist()
            sql_query = generate_sql(query, dataset, column_names)
            
            # Display SQL query
            st.subheader("Generated SQL Query")
            st.code(sql_query)
            
        
    else:
        st.error("Error loading dataset")
