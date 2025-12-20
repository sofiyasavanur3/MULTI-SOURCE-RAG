"""
Unified Ingestion Manager

WHY THIS EXISTS:
- Single interface for all data sources (PDFs, Web, CSV)
- Handles duplicate detection across sources
- Provides unified metadata and tracking
- Simplifies the ingestion process

HOW IT WORKS:
1. Initialize with all ingesters
2. Add sources from different types
3. Deduplicate content
4. Return unified document list

BENEFITS:
- One API for everything
- Automatic deduplication
- Source tracking
- Easy to extend with new sources
"""

from typing import List, Dict, Optional
from llama_index.core import Document
from pathlib import Path
import logging
from datetime import datetime
import hashlib

# Import our custom ingesters
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_ingester import PDFIngester
from src.ingestion.web_scraper import WebScraper
from src.ingestion.csv_ingester import CSVIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedIngestionManager:
    """
    Manages ingestion from multiple data sources
    
    WHY USE A CLASS?
    - Maintains state (all documents in one place)
    - Coordinates multiple ingesters
    - Handles deduplication
    - Provides unified interface
    
    ANALOGY:
    Like a project manager who coordinates different teams
    (PDF team, Web team, CSV team) and combines their work
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        web_delay: float = 1.0
    ):
        """
        Initialize unified manager
        
        Parameters:
        - chunk_size: Size of text chunks (default 1024)
        - chunk_overlap: Overlap between chunks (default 200)
        - web_delay: Delay between web requests (default 1.0s)
        
        WHY THESE PARAMETERS?
        - Apply same chunking across all sources (consistency!)
        - Control web scraping speed
        """
        # Initialize all ingesters
        # WHY? Ready to handle any source type
        self.pdf_ingester = PDFIngester(chunk_size, chunk_overlap)
        self.web_scraper = WebScraper(timeout=10, delay=web_delay)
        self.csv_ingester = CSVIngester()
        
        # Store all documents
        # WHY LIST? Easy to add, iterate, and index
        self.documents: List[Document] = []
        
        # Track sources
        # WHY? Know what we've ingested, avoid duplicates
        self.sources_added: Dict[str, List[str]] = {
            'pdfs': [],
            'urls': [],
            'csvs': []
        }
        
        # Track document hashes for deduplication
        # WHY? Same content might come from multiple sources
        self.content_hashes: set = set()
        
        logger.info("🎯 Unified Ingestion Manager initialized")
        logger.info(f"   • Chunk size: {chunk_size}")
        logger.info(f"   • Chunk overlap: {chunk_overlap}")
        logger.info(f"   • Web delay: {web_delay}s")
    
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute hash of text for deduplication
        
        WHY HASHING?
        - Quick comparison (hash is shorter than full text)
        - Detects exact duplicates
        - Memory efficient
        
        HOW IT WORKS:
        Text → SHA256 → "a3f2b1c4..." (unique fingerprint)
        
        EXAMPLE:
        "Hello World" → "a591a6d40b..."
        "Hello World" → "a591a6d40b..." (same hash!)
        "Hello World!" → "different hash" (different text)
        
        Returns: Hexadecimal hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    
    def _is_duplicate(self, text: str) -> bool:
        """
        Check if content is duplicate
        
        WHY?
        - Same document might be in multiple places
        - Avoid indexing same content twice
        - Save processing time and storage
        
        EXAMPLE:
        Company policy PDF on website AND in local folder
        → Only index once!
        
        Returns: True if duplicate, False if new
        """
        content_hash = self._compute_hash(text)
        
        if content_hash in self.content_hashes:
            logger.debug(f"⚠️  Duplicate content detected (hash: {content_hash[:16]}...)")
            return True
        
        # Add to seen hashes
        self.content_hashes.add(content_hash)
        return False
    
    
    def add_pdfs(self, directory_path: str) -> int:
        """
        Add PDFs from a directory
        
        WHY SEPARATE METHOD?
        - Clear API: manager.add_pdfs(...)
        - Can be called multiple times for different folders
        - Returns count for user feedback
        
        Returns: Number of documents added
        """
        logger.info(f"📄 Adding PDFs from: {directory_path}")
        
        try:
            # Ingest PDFs
            pdf_documents = self.pdf_ingester.ingest_directory(directory_path)
            
            # Chunk them
            # WHY? PDFs can be long, need smaller pieces
            chunks = self.pdf_ingester.chunk_documents(pdf_documents)
            
            # Add non-duplicate chunks
            added_count = 0
            for chunk in chunks:
                if not self._is_duplicate(chunk.text):
                    self.documents.append(chunk)
                    added_count += 1
            
            # Track this source
            self.sources_added['pdfs'].append(directory_path)
            
            logger.info(f"✅ Added {added_count} PDF chunks ({len(chunks) - added_count} duplicates skipped)")
            return added_count
            
        except Exception as e:
            logger.error(f"❌ Error adding PDFs: {str(e)}")
            return 0
    
    
    def add_urls(self, urls: List[str]) -> int:
        """
        Add web pages from URLs
        
        WHY?
        - Batch processing of multiple URLs
        - Automatic deduplication
        - Error handling per URL
        
        Returns: Number of documents added
        """
        logger.info(f"🌐 Adding {len(urls)} URL(s)")
        
        try:
            # Scrape URLs
            web_documents = self.web_scraper.scrape_urls(urls)
            
            # Add non-duplicate documents
            added_count = 0
            for doc in web_documents:
                if not self._is_duplicate(doc.text):
                    self.documents.append(doc)
                    added_count += 1
            
            # Track sources
            self.sources_added['urls'].extend(urls)
            
            logger.info(f"✅ Added {added_count} web documents ({len(web_documents) - added_count} duplicates skipped)")
            return added_count
            
        except Exception as e:
            logger.error(f"❌ Error adding URLs: {str(e)}")
            return 0
    
    
    def add_csvs(self, directory_path: str) -> int:
        """
        Add CSV files from a directory
        
        WHY?
        - Structured data needs special handling
        - Converts tables to searchable text
        - Preserves relationships
        
        Returns: Number of documents added
        """
        logger.info(f"📊 Adding CSVs from: {directory_path}")
        
        try:
            # Ingest CSVs
            csv_documents = self.csv_ingester.ingest_directory(directory_path)
            
            # Add non-duplicate documents
            added_count = 0
            for doc in csv_documents:
                if not self._is_duplicate(doc.text):
                    self.documents.append(doc)
                    added_count += 1
            
            # Track sources
            self.sources_added['csvs'].append(directory_path)
            
            logger.info(f"✅ Added {added_count} CSV documents ({len(csv_documents) - added_count} duplicates skipped)")
            return added_count
            
        except Exception as e:
            logger.error(f"❌ Error adding CSVs: {str(e)}")
            return 0
    
    
    def add_all_sources(
        self,
        pdf_dir: str = "data/pdfs",
        csv_dir: str = "data/databases",
        urls: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Convenience method to add all sources at once
        
        WHY?
        - One call does everything!
        - Perfect for initialization
        - Returns summary of what was added
        
        EXAMPLE:
        manager.add_all_sources()
        → Ingests everything from default locations
        
        Returns: Dictionary with counts per source type
        """
        logger.info("🚀 Adding all sources...")
        
        counts = {
            'pdfs': 0,
            'urls': 0,
            'csvs': 0
        }
        
        # Add PDFs if directory exists
        # WHY CHECK? Don't crash if folder doesn't exist
        if Path(pdf_dir).exists():
            counts['pdfs'] = self.add_pdfs(pdf_dir)
        else:
            logger.warning(f"⚠️  PDF directory not found: {pdf_dir}")
        
        # Add CSVs if directory exists
        if Path(csv_dir).exists():
            counts['csvs'] = self.add_csvs(csv_dir)
        else:
            logger.warning(f"⚠️  CSV directory not found: {csv_dir}")
        
        # Add URLs if provided
        if urls:
            counts['urls'] = self.add_urls(urls)
        
        logger.info(f"✅ Total documents added: {sum(counts.values())}")
        return counts
    
    
    def get_documents(self) -> List[Document]:
        """
        Get all ingested documents
        
        WHY?
        - Main output of the manager
        - Ready for indexing
        - Deduplicated and processed
        
        Returns: List of all documents
        """
        return self.documents
    
    
    def get_statistics(self) -> Dict:
        """
        Get ingestion statistics
        
        WHY?
        - User feedback
        - Debugging
        - Monitoring
        
        Returns: Dictionary with statistics
        """
        # Count documents by source type
        # WHY? Know distribution of content
        source_counts = {
            'pdf': 0,
            'web': 0,
            'csv': 0
        }
        
        for doc in self.documents:
            source_type = doc.metadata.get('source_type', 'unknown')
            if source_type in source_counts:
                source_counts[source_type] += 1
        
        # Calculate total text length
        # WHY? Know how much content we have
        total_chars = sum(len(doc.text) for doc in self.documents)
        
        stats = {
            'total_documents': len(self.documents),
            'by_source': source_counts,
            'total_characters': total_chars,
            'average_doc_length': total_chars // len(self.documents) if self.documents else 0,
            'sources_added': self.sources_added,
            'unique_content_hashes': len(self.content_hashes)
        }
        
        return stats
    
    
    def clear(self):
        """
        Clear all documents and reset
        
        WHY?
        - Fresh start without creating new instance
        - Useful for testing
        - Memory cleanup
        """
        logger.info("🧹 Clearing all documents")
        self.documents = []
        self.content_hashes = set()
        self.sources_added = {
            'pdfs': [],
            'urls': [],
            'csvs': []
        }
    
    
    def export_summary(self, output_file: str = "ingestion_summary.txt"):
        """
        Export a summary of ingested content
        
        WHY?
        - Documentation
        - Audit trail
        - Share with team
        
        Creates: Text file with summary
        """
        stats = self.get_statistics()
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("INGESTION SUMMARY\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Documents: {stats['total_documents']}\n")
            f.write(f"Total Characters: {stats['total_characters']:,}\n")
            f.write(f"Average Document Length: {stats['average_doc_length']} chars\n\n")
            
            f.write("Documents by Source:\n")
            for source, count in stats['by_source'].items():
                f.write(f"  • {source.upper()}: {count}\n")
            
            f.write("\nSources Added:\n")
            for source_type, paths in stats['sources_added'].items():
                f.write(f"  • {source_type.upper()}:\n")
                for path in paths:
                    f.write(f"    - {path}\n")
            
            f.write(f"\nUnique Content Pieces: {stats['unique_content_hashes']}\n")
        
        logger.info(f"📄 Summary exported to: {output_file}")


def demo_unified_manager():
    """
    Demonstrates the unified manager
    
    WHY?
    - Shows complete workflow
    - Tests all components together
    - Educational example
    """
    print("\n" + "="*60)
    print("🎯 UNIFIED INGESTION MANAGER DEMO")
    print("="*60 + "\n")
    
    # Initialize manager
    manager = UnifiedIngestionManager(
        chunk_size=1024,
        chunk_overlap=200,
        web_delay=1.0
    )
    
    # Add all sources
    print("📥 Ingesting from all sources...\n")
    
    # Example URLs (optional)
    # WHY OPTIONAL? User might not want to scrape web every time
    example_urls = [
        "https://www.llamaindex.ai/",
    ]
    
    counts = manager.add_all_sources(urls=example_urls)
    
    # Display results
    print("\n" + "="*60)
    print("📊 INGESTION RESULTS")
    print("="*60)
    print(f"\n✅ Documents Added:")
    print(f"   • PDFs: {counts['pdfs']}")
    print(f"   • Web pages: {counts['urls']}")
    print(f"   • CSV rows: {counts['csvs']}")
    print(f"   • TOTAL: {sum(counts.values())}")
    
    # Get statistics
    stats = manager.get_statistics()
    
    print(f"\n📈 STATISTICS:")
    print(f"   • Total documents: {stats['total_documents']}")
    print(f"   • Total characters: {stats['total_characters']:,}")
    print(f"   • Average length: {stats['average_doc_length']} chars")
    print(f"   • Unique content pieces: {stats['unique_content_hashes']}")
    
    # Show sample document from each source
    print(f"\n📝 SAMPLE DOCUMENTS:")
    
    seen_types = set()
    for doc in manager.get_documents():
        source_type = doc.metadata.get('source_type')
        if source_type not in seen_types and source_type:
            seen_types.add(source_type)
            print(f"\n   [{source_type.upper()}] Sample:")
            print(f"   Source: {doc.metadata.get('filename') or doc.metadata.get('url') or doc.metadata.get('source_file')}")
            print(f"   Preview: {doc.text[:150]}...")
    
    # Export summary
    manager.export_summary("ingestion_summary.txt")
    print(f"\n💾 Summary exported to: ingestion_summary.txt")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_unified_manager()