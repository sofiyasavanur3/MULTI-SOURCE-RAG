"""
PDF Ingestion Module

WHY THIS EXISTS:
- Converts PDFs into searchable text chunks
- Preserves metadata (filename, page numbers) for citations
- Uses smart chunking to maintain context

HOW IT WORKS:
1. Read PDF files from a directory
2. Extract text page by page
3. Split into chunks (not too small, not too large)
4. Add metadata for source tracking
"""

import os
from typing import List
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from pypdf import PdfReader
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFIngester:
    """
    Handles PDF ingestion with metadata preservation
    
    WHY USE A CLASS?
    - Keeps all PDF-related functions together
    - Easy to reuse and maintain
    - Can store configuration (chunk size, etc.)
    """
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Initialize the PDF ingester
        
        Parameters:
        - chunk_size: How many characters per chunk (default 1024)
          WHY? Too small = loses context, too large = irrelevant info
        - chunk_overlap: How many characters overlap between chunks (default 200)
          WHY? Ensures we don't split important sentences/paragraphs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Node parser splits documents into chunks
        # WHY? LLMs have token limits, need smaller pieces
        self.parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"📄 PDF Ingester initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[dict]:
        """
        Extract text from a single PDF file
        
        WHY SEPARATE FUNCTION?
        - Focuses on ONE job: PDF → Text
        - Easy to test individually
        - Can handle errors for specific files
        
        Returns: List of dictionaries with text and metadata
        """
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            documents = []
            
            # Process each page separately
            # WHY? Preserves page numbers for citations
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Skip empty pages
                # WHY? No point indexing blank pages
                if not text.strip():
                    continue
                
                # Create document with metadata
                # WHY METADATA? For citations: "Found on page 5 of report.pdf"
                documents.append({
                    'text': text,
                    'metadata': {
                        'filename': filename,
                        'page': page_num,
                        'source_type': 'pdf',
                        'total_pages': len(reader.pages)
                    }
                })
            
            logger.info(f"✅ Extracted {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {str(e)}")
            return []
    
    
    def ingest_directory(self, directory_path: str) -> List[Document]:
        """
        Ingest all PDFs from a directory
        
        WHY?
        - Users often have multiple PDFs
        - Batch processing is more efficient
        - One function call to process everything
        
        Returns: List of LlamaIndex Document objects
        """
        pdf_dir = Path(directory_path)
        
        # Check if directory exists
        # WHY? Prevent crashes from typos
        if not pdf_dir.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all PDF files
        # WHY *.pdf? Only process PDF files, ignore images, text files, etc.
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {directory_path}")
            return []
        
        logger.info(f"📚 Found {len(pdf_files)} PDF file(s)")
        
        all_documents = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            logger.info(f"📄 Processing: {pdf_file.name}")
            
            # Extract text from this PDF
            pages = self.extract_text_from_pdf(str(pdf_file))
            
            # Convert to LlamaIndex Document format
            # WHY? LlamaIndex needs this specific format
            for page_data in pages:
                doc = Document(
                    text=page_data['text'],
                    metadata=page_data['metadata']
                )
                all_documents.append(doc)
        
        logger.info(f"✅ Total documents created: {len(all_documents)}")
        return all_documents
    
    
    def chunk_documents(self, documents: List[Document]) -> List:
        """
        Split documents into smaller chunks
        
        WHY CHUNKING?
        1. LLMs have token limits (can't process entire books)
        2. Smaller chunks = more precise retrieval
        3. Better context matching
        
        EXAMPLE:
        Before: "Chapter 1: Introduction... [5000 words]"
        After: ["Chapter 1: Introduction... [500 words]", 
                "The main concept is... [500 words]", ...]
        """
        logger.info(f"✂️  Chunking {len(documents)} documents...")
        nodes = self.parser.get_nodes_from_documents(documents)
        logger.info(f"✅ Created {len(nodes)} chunks")
        return nodes


# Example usage function
def demo_pdf_ingestion():
    """
    Demonstrates how to use the PDF ingester
    
    WHY DEMO FUNCTION?
    - Shows developers how to use the class
    - Easy to test if everything works
    - Can be run independently
    """
    print("\n" + "="*60)
    print("📄 PDF INGESTION DEMO")
    print("="*60 + "\n")
    
    # Initialize ingester
    ingester = PDFIngester(chunk_size=1024, chunk_overlap=200)
    
    # Ingest PDFs
    pdf_directory = "data/pdfs"
    
    # Check if directory exists, create if not
    if not os.path.exists(pdf_directory):
        print(f"Creating directory: {pdf_directory}")
        os.makedirs(pdf_directory)
        print(f"⚠️  Please add PDF files to {pdf_directory} and run again!")
        return
    
    try:
        # Process all PDFs
        documents = ingester.ingest_directory(pdf_directory)
        
        if documents:
            # Chunk them
            chunks = ingester.chunk_documents(documents)
            
            print(f"\n📊 RESULTS:")
            print(f"   • Total documents: {len(documents)}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Average chunk size: {sum(len(chunk.text) for chunk in chunks) // len(chunks)} characters")
            
            # Show a sample chunk
            if chunks:
                print(f"\n📝 SAMPLE CHUNK:")
                print(f"   Source: {chunks[0].metadata.get('filename', 'Unknown')}")
                print(f"   Page: {chunks[0].metadata.get('page', 'Unknown')}")
                print(f"   Text preview: {chunks[0].text[:200]}...")
        else:
            print("⚠️  No documents were processed. Add PDF files to data/pdfs/")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_pdf_ingestion()