"""
Web Scraping Module

WHY THIS EXISTS:
- Extracts clean text from websites
- Handles multiple URLs
- Removes ads, navigation, footers (keeps only content)
- Preserves source URL for citations

HOW IT WORKS:
1. Fetch webpage HTML
2. Parse and clean (remove scripts, ads)
3. Extract main content
4. Store with metadata (URL, title, date scraped)
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from llama_index.core import Document
from urllib.parse import urlparse
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    Scrapes and cleans web content for RAG ingestion
    
    WHY USE A CLASS?
    - Configure once (headers, timeout), use many times
    - Maintains session (faster for multiple requests)
    - Easy to add rate limiting (don't overwhelm servers)
    """
    
    def __init__(self, timeout: int = 10, delay: float = 1.0):
        """
        Initialize web scraper
        
        Parameters:
        - timeout: Seconds to wait for response (default 10)
          WHY? Some sites are slow, but don't wait forever
        - delay: Seconds between requests (default 1.0)
          WHY? Polite scraping, don't overwhelm servers
        """
        self.timeout = timeout
        self.delay = delay
        
        # Headers make us look like a real browser
        # WHY? Some sites block "bots", this helps us appear legitimate
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        logger.info(f"🌐 Web Scraper initialized (timeout={timeout}s, delay={delay}s)")
    
    
    def fetch_url(self, url: str) -> str:
        """
        Fetch HTML content from a URL
        
        WHY SEPARATE FUNCTION?
        - Handles errors (404, timeout) gracefully
        - Adds delays between requests
        - Easy to add caching later
        
        Returns: HTML string or empty string on error
        """
        try:
            logger.info(f"📡 Fetching: {url}")
            
            # Make the request
            # WHY headers? To look like a real browser
            # WHY timeout? Don't wait forever for slow sites
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            
            # Check if successful
            # WHY? 404 = page not found, 403 = blocked, etc.
            response.raise_for_status()
            
            # Be polite - wait between requests
            # WHY? Prevents overwhelming servers, avoids getting blocked
            time.sleep(self.delay)
            
            logger.info(f"✅ Successfully fetched {url}")
            return response.text
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️  Timeout fetching {url}")
            return ""
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP error for {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Error fetching {url}: {str(e)}")
            return ""
    
    
    def clean_html(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract clean text from HTML
        
        WHY CLEANING?
        - HTML has <script>, <style>, navigation, ads, footers
        - We only want the MAIN CONTENT
        - Clean text = better search results
        
        EXAMPLE:
        Before: <div><script>ads()</script><p>Content</p></div>
        After: "Content"
        
        Returns: Dict with title, text, url
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        # WHY? These don't contain useful content
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Extract title
        # WHY? Good for citations: "Found in: Guide to AI (www.example.com)"
        title = soup.title.string if soup.title else urlparse(url).path
        title = title.strip() if title else "Untitled"
        
        # Extract main content
        # WHY? <main>, <article> tags usually contain the good stuff
        main_content = soup.find('main') or soup.find('article') or soup.body
        
        if main_content:
            # Get text and clean it
            text = main_content.get_text(separator=' ', strip=True)
            
            # Remove extra whitespace
            # WHY? HTML often has tons of spaces/newlines
            text = ' '.join(text.split())
        else:
            text = ""
        
        logger.info(f"📝 Extracted {len(text)} characters from {url}")
        
        return {
            'title': title,
            'text': text,
            'url': url
        }
    
    
    def scrape_url(self, url: str) -> Document:
        """
        Scrape a single URL and convert to Document
        
        WHY?
        - Combines fetch + clean into one step
        - Returns LlamaIndex Document format
        - Adds complete metadata
        
        Returns: LlamaIndex Document with metadata
        """
        # Fetch HTML
        html = self.fetch_url(url)
        
        if not html:
            logger.warning(f"⚠️  No content from {url}")
            return None
        
        # Clean and extract
        cleaned = self.clean_html(html, url)
        
        if not cleaned['text']:
            logger.warning(f"⚠️  No text extracted from {url}")
            return None
        
        # Create Document with rich metadata
        # WHY METADATA? For citations and source tracking
        doc = Document(
            text=cleaned['text'],
            metadata={
                'title': cleaned['title'],
                'url': url,
                'source_type': 'web',
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(cleaned['text'].split())
            }
        )
        
        logger.info(f"✅ Created document from {url}")
        return doc
    
    
    def scrape_urls(self, urls: List[str]) -> List[Document]:
        """
        Scrape multiple URLs
        
        WHY?
        - Users often have multiple pages to scrape
        - Batch processing with progress tracking
        - Continues even if some URLs fail
        
        Returns: List of Documents (skips failed URLs)
        """
        logger.info(f"🌐 Scraping {len(urls)} URL(s)...")
        
        documents = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"📄 [{i}/{len(urls)}] Processing {url}")
            
            doc = self.scrape_url(url)
            
            if doc:
                documents.append(doc)
        
        logger.info(f"✅ Successfully scraped {len(documents)}/{len(urls)} URLs")
        return documents


def demo_web_scraping():
    """
    Demonstrates web scraping functionality
    
    WHY DEMO?
    - Shows how to use the scraper
    - Easy to test with real URLs
    - Educational for other developers
    """
    print("\n" + "="*60)
    print("🌐 WEB SCRAPING DEMO")
    print("="*60 + "\n")
    
    # Initialize scraper
    scraper = WebScraper(timeout=10, delay=1.0)
    
    # Example URLs (you can change these!)
    # WHY THESE? They're documentation sites (publicly scrapable)
    urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://www.llamaindex.ai/",
    ]
    
    print("📋 URLs to scrape:")
    for url in urls:
        print(f"   • {url}")
    print()
    
    # Scrape the URLs
    documents = scraper.scrape_urls(urls)
    
    # Display results
    if documents:
        print(f"\n📊 RESULTS:")
        print(f"   • Successfully scraped: {len(documents)} page(s)")
        print()
        
        for i, doc in enumerate(documents, 1):
            print(f"📄 Document {i}:")
            print(f"   Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"   URL: {doc.metadata.get('url', 'Unknown')}")
            print(f"   Words: {doc.metadata.get('word_count', 0)}")
            print(f"   Preview: {doc.text[:150]}...")
            print()
    else:
        print("⚠️  No documents were created")
    
    print("="*60)


if __name__ == "__main__":
    demo_web_scraping()