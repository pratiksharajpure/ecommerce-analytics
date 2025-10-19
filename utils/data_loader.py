import pandas as pd
import streamlit as st
from pathlib import Path

def load_csv(csv_name, folder="core_data"):
    """Load CSV file from sample_data directory"""
    try:
        file_path = Path(f"sample_data/{folder}/{csv_name}")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_cached_csv(csv_name, folder="core_data"):
    """Load CSV with caching (1 hour TTL)"""
    return load_csv(csv_name, folder)