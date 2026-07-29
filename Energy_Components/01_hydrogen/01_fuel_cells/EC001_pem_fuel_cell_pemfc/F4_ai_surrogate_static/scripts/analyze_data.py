import fitz
import pandas as pd

def extract_pdf_abstract(pdf_path, text_output_path):
    print(f"Reading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Extracted {len(text)} characters from {len(doc)} pages.")
    print("First 1000 characters:")
    print(text[:1000])
    
def analyze_dataset(parquet_path):
    print(f"\nReading Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nDescribe:")
    print(df.describe().T)
    print("\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    extract_pdf_abstract("data/main resource.pdf", "pdf_text.txt")
    analyze_dataset("data/training.parquet")
