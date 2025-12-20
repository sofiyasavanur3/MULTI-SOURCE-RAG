"""
CSV/Database Ingestion Module

WHY THIS EXISTS:
- Converts structured data (tables) into searchable text
- Handles various CSV formats and encodings
- Creates meaningful text representations of rows
- Preserves data types and relationships

HOW IT WORKS:
1. Read CSV file
2. Convert each row to a natural language sentence
3. Add metadata (column names, data types, source)
4. Create searchable documents

EXAMPLE:
CSV Row: {name: "Priya", role: "CEO", salary: 200000}
Converted: "Priya is the CEO with a salary of 200000"
"""

import pandas as pd
from typing import List, Dict, Optional
from llama_index.core import Document
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVIngester:
    """
    Ingests structured data from CSV files
    
    WHY USE A CLASS?
    - Can handle multiple CSV files with same configuration
    - Maintains state (loaded dataframes, schemas)
    - Easy to extend for SQL databases later
    """
    
    def __init__(self, text_columns: Optional[List[str]] = None):
        """
        Initialize CSV ingester
        
        Parameters:
        - text_columns: Specific columns to focus on (optional)
          WHY? Some columns are more important (description vs id)
        """
        self.text_columns = text_columns
        self.dataframes = {}  # Store loaded CSVs
        
        logger.info("📊 CSV Ingester initialized")
    
    
    def load_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load CSV file with error handling
        
        WHY SEPARATE FUNCTION?
        - Handles different encodings (UTF-8, Latin1, etc.)
        - Validates file exists
        - Catches parsing errors
        
        WHY ENCODING PARAMETER?
        - CSVs from Excel might be Latin1
        - CSVs from different countries have different encodings
        - Wrong encoding = garbled text
        
        Returns: Pandas DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            logger.info(f"📂 Loading CSV: {file_path.name}")
            
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning(f"⚠️  Encoding '{encoding}' failed, trying 'latin1'")
                df = pd.read_csv(file_path, encoding='latin1')
            
            self.dataframes[file_path.name] = df
            
            logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"📋 Columns: {', '.join(df.columns.tolist())}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {str(e)}")
            raise
    
    
    def row_to_text(self, row: pd.Series, columns: Optional[List[str]] = None) -> str:
        """
        Convert a CSV row to natural language text
        
        WHY?
        - RAG needs TEXT, not structured data
        - Creates searchable, meaningful sentences
        - Preserves all information in readable format
        
        Returns: Natural language string
        """
        cols = columns or row.index.tolist()
        
        text_parts = []
        
        for col in cols:
            value = row[col]
            
            if pd.isna(value) or value == '':
                continue
            
            text_parts.append(f"{col} is {value}")
        
        return ". ".join(text_parts) + "."
    
    
    def dataframe_to_documents(
        self, 
        df: pd.DataFrame, 
        source_name: str,
        columns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Convert entire DataFrame to LlamaIndex Documents
        
        Returns: List of LlamaIndex Documents
        """
        logger.info(f"🔄 Converting {len(df)} rows to documents...")
        
        documents = []
        
        for idx, row in df.iterrows():
            text = self.row_to_text(row, columns)
            
            if not text.strip():
                continue
            
            doc = Document(
                text=text,
                metadata={
                    'source_file': source_name,
                    'source_type': 'csv',
                    'row_number': int(idx),
                    'columns': list(df.columns),
                    'ingested_at': datetime.now().isoformat(),
                    'row_data': row.to_dict()
                }
            )
            
            documents.append(doc)
        
        logger.info(f"✅ Created {len(documents)} documents from DataFrame")
        return documents
    
    
    def ingest_csv(
        self, 
        file_path: str, 
        columns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Complete CSV ingestion pipeline
        
        Returns: List of Documents ready for indexing
        """
        df = self.load_csv(file_path)
        
        source_name = Path(file_path).name
        documents = self.dataframe_to_documents(df, source_name, columns)
        
        return documents
    
    
    def ingest_directory(self, directory_path: str) -> List[Document]:
        """
        Ingest all CSV files from a directory
        
        Returns: Combined list of all documents
        """
        csv_dir = Path(directory_path)
        
        if not csv_dir.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"⚠️  No CSV files found in {directory_path}")
            return []
        
        logger.info(f"📚 Found {len(csv_files)} CSV file(s)")
        
        all_documents = []
        
        for csv_file in csv_files:
            try:
                logger.info(f"📊 Processing: {csv_file.name}")
                documents = self.ingest_csv(str(csv_file))
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"❌ Failed to process {csv_file.name}: {str(e)}")
                continue
        
        logger.info(f"✅ Total documents from all CSVs: {len(all_documents)}")
        return all_documents
    
    
    def get_schema_info(self, file_path: str) -> Dict:
        """
        Get schema information about a CSV
        
        Returns: Dictionary with schema info
        """
        df = self.load_csv(file_path)
        
        schema = {
            'filename': Path(file_path).name,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_row': df.iloc[0].to_dict() if len(df) > 0 else {}
        }
        
        return schema


def demo_csv_ingestion():
    """Demonstrates CSV ingestion functionality"""
    print("\n" + "="*60)
    print("📊 CSV INGESTION DEMO")
    print("="*60 + "\n")
    
    sample_csv_path = "data/databases/employees.csv"
    
    Path("data/databases").mkdir(parents=True, exist_ok=True)
    
    sample_data = pd.DataFrame({
        'Name': ['Priya Sharma', 'Raj Kumar', 'Anita Desai', 'Vikram Singh'],
        'Role': ['CEO', 'CTO', 'CFO', 'VP Engineering'],
        'Department': ['Executive', 'Engineering', 'Finance', 'Engineering'],
        'Salary': [200000, 180000, 190000, 160000],
        'Years': [5, 4, 3, 6],
        'Location': ['Bangalore', 'Bangalore', 'Mumbai', 'Delhi']
    })
    
    sample_data.to_csv(sample_csv_path, index=False)
    logger.info(f"✅ Created sample CSV: {sample_csv_path}")
    
    ingester = CSVIngester()
    
    print("\n📋 CSV SCHEMA:")
    schema = ingester.get_schema_info(sample_csv_path)
    print(f"   • File: {schema['filename']}")
    print(f"   • Rows: {schema['num_rows']}")
    print(f"   • Columns: {schema['num_columns']}")
    print(f"   • Column names: {', '.join(schema['columns'])}")
    
    print("\n🔄 INGESTING CSV...")
    documents = ingester.ingest_csv(sample_csv_path)
    
    if documents:
        print(f"\n📊 RESULTS:")
        print(f"   • Total documents: {len(documents)}")
        print("\n📝 SAMPLE DOCUMENTS:")
        for i, doc in enumerate(documents[:2], 1):
            print(f"\n   Document {i}:")
            print(f"   Source: {doc.metadata.get('source_file')}")
            print(f"   Row: {doc.metadata.get('row_number')}")
            print(f"   Text: {doc.text}")
    else:
        print("⚠️  No documents were created")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_csv_ingestion()