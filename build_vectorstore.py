import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ modern import
from langchain_core.documents import Document
import os

# Paths
csv_path = "data/income_tax_act_2024_sections.csv"
index_path = "data/faiss_index"

# Load CSV
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Text"])
df = df[df["Text"].str.strip().astype(bool)]

# Convert to Document objects
docs = [
    Document(page_content=str(row["Text"]), metadata={"section": str(row["Section"])})
    for _, row in df.iterrows()
]

# ✅ Create embeddings (offline, free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save locally
db.save_local(index_path)
print(f"✅ FAISS index built and saved to {index_path}")
