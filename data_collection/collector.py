"""
Data Collection Module

This module is responsible for collecting data from various sources:
- Research papers from arXiv
- Technical blog posts
- Social media content
- Industry reports
"""

import logging
import datetime
from typing import Dict, List, Any, Optional
import requests
import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects data from various sources for content generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data collector with configuration.
        
        Args:
            config: Configuration dictionary for data collection
        """
        self.config = config
        self.sources = config.get('sources', {})
        self.update_frequency = config.get('update_frequency', {})
        logger.info("Data collector initialized")
    
    def collect(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from all configured sources.
        
        Returns:
            Dictionary containing collected data from each source
        """
        results = {}
        
        # Collect from arXiv if enabled
        if self.sources.get('arxiv', {}).get('enabled', False):
            results['arxiv'] = self._collect_from_arxiv()
        
        # Collect from technical blogs if enabled
        if self.sources.get('technical_blogs', {}).get('enabled', False):
            results['technical_blogs'] = self._collect_from_technical_blogs()
        
        # Collect from social media if enabled
        if self.sources.get('social_media', {}).get('enabled', False):
            results['social_media'] = self._collect_from_social_media()
        
        # Collect from industry reports if enabled
        if self.sources.get('industry_reports', {}).get('enabled', False):
            results['industry_reports'] = self._collect_from_industry_reports()
        
        logger.info(f"Collected data from {len(results)} sources")
        return results
    
    def _collect_from_arxiv(self) -> List[Dict[str, Any]]:
        """
        Collect research papers from arXiv.
        
        Returns:
            List of papers with metadata
        """
        logger.info("Collecting data from arXiv")
        
        arxiv_config = self.sources.get('arxiv', {})
        categories = arxiv_config.get('categories', ['cs.AI', 'cs.LG'])
        max_results = arxiv_config.get('max_results', 100)
        days_back = arxiv_config.get('days_back', 30)
        
        # Calculate date for filtering
        start_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        
        # Prepare query
        category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
        query = f'({category_query}) AND submittedDate:[{start_date.strftime("%Y%m%d")}000000 TO now]'
        
        # Prepare API URL
        url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
        
        try:
            # Parse the feed
            feed = feedparser.parse(url)
            
            # Extract paper information
            papers = []
            for entry in feed.entries:
                paper = {
                    'id': entry.id.split('/')[-1],
                    'title': entry.title,
                    'authors': [author.name for author in entry.authors],
                    'summary': entry.summary,
                    'published': entry.published,
                    'link': entry.link,
                    'pdf_link': next((link.href for link in entry.links if link.rel == 'alternate' and link.type == 'application/pdf'), None),
                    'categories': [tag.term for tag in entry.tags],
                }
                papers.append(paper)
            
            logger.info(f"Collected {len(papers)} papers from arXiv")
            return papers
        
        except Exception as e:
            logger.error(f"Error collecting data from arXiv: {e}")
            return []
    
    def _collect_from_technical_blogs(self) -> List[Dict[str, Any]]:
        """
        Collect posts from technical blogs.
        
        Returns:
            List of blog posts with metadata
        """
        logger.info("Collecting data from technical blogs")
        
        blogs_config = self.sources.get('technical_blogs', {})
        sources = blogs_config.get('sources', [])
        max_posts_per_source = blogs_config.get('max_posts_per_source', 10)
        
        all_posts = []
        
        for source in sources:
            try:
                name = source.get('name', 'Unknown')
                url = source.get('url', '')
                
                logger.info(f"Collecting from {name} ({url})")
                
                # Try to find RSS feed first
                rss_url = self._discover_rss_feed(url)
                
                if rss_url:
                    # Parse the feed
                    feed = feedparser.parse(rss_url)
                    
                    # Extract post information
                    posts = []
                    for entry in feed.entries[:max_posts_per_source]:
                        post = {
                            'title': entry.title,
                            'summary': entry.get('summary', ''),
                            'content': entry.get('content', [{'value': ''}])[0].get('value', ''),
                            'published': entry.get('published', ''),
                            'link': entry.get('link', ''),
                            'source': name,
                            'source_url': url,
                        }
                        posts.append(post)
                else:
                    # Fallback to web scraping
                    posts = self._scrape_blog_posts(url, name, max_posts_per_source)
                
                all_posts.extend(posts)
                logger.info(f"Collected {len(posts)} posts from {name}")
                
            except Exception as e:
                logger.error(f"Error collecting data from {source.get('name', 'Unknown')}: {e}")
        
        logger.info(f"Collected {len(all_posts)} posts from technical blogs")
        return all_posts
    
    def _discover_rss_feed(self, url: str) -> Optional[str]:
        """
        Discover RSS feed URL for a given website.
        
        Args:
            url: Website URL
            
        Returns:
            RSS feed URL if found, None otherwise
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for RSS link
            rss_link = soup.find('link', type='application/rss+xml')
            if rss_link and 'href' in rss_link.attrs:
                rss_url = rss_link['href']
                
                # Handle relative URLs
                if rss_url.startswith('/'):
                    rss_url = f"{url.rstrip('/')}{rss_url}"
                
                return rss_url
            
            return None
        
        except Exception as e:
            logger.error(f"Error discovering RSS feed for {url}: {e}")
            return None
    
    def _scrape_blog_posts(self, url: str, source_name: str, max_posts: int) -> List[Dict[str, Any]]:
        """
        Scrape blog posts from a website.
        
        Args:
            url: Website URL
            source_name: Name of the source
            max_posts: Maximum number of posts to scrape
            
        Returns:
            List of scraped blog posts
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a simplified implementation and may need to be customized for each blog
            # In a real implementation, you would need site-specific selectors
            
            posts = []
            
            # Look for article elements (common in blog layouts)
            articles = soup.find_all('article')[:max_posts]
            
            for article in articles:
                # Extract title
                title_elem = article.find(['h1', 'h2', 'h3'])
                title = title_elem.text.strip() if title_elem else 'Unknown Title'
                
                # Extract link
                link_elem = title_elem.find('a') if title_elem else None
                link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else ''
                
                # Handle relative URLs
                if link.startswith('/'):
                    link = f"{url.rstrip('/')}{link}"
                
                # Extract summary
                summary_elem = article.find(['p', 'div'], class_=['summary', 'excerpt', 'description'])
                summary = summary_elem.text.strip() if summary_elem else ''
                
                # Extract date
                date_elem = article.find(['time', 'span'], class_=['date', 'published', 'time'])
                published = date_elem.text.strip() if date_elem else ''
                
                post = {
                    'title': title,
                    'summary': summary,
                    'content': '',  # Would require visiting each post URL
                    'published': published,
                    'link': link,
                    'source': source_name,
                    'source_url': url,
                }
                
                posts.append(post)
            
            return posts
        
        except Exception as e:
            logger.error(f"Error scraping blog posts from {url}: {e}")
            return []
    
    def _collect_from_social_media(self) -> List[Dict[str, Any]]:
        """
        Collect content from social media platforms.
        
        Returns:
            List of social media posts with metadata
        """
        logger.info("Collecting data from social media")
        
        social_config = self.sources.get('social_media', {})
        platforms = social_config.get('platforms', {})
        
        all_posts = []
        
        # Collect from Twitter if enabled
        if platforms.get('twitter', {}).get('enabled', False):
            twitter_posts = self._collect_from_twitter(platforms.get('twitter', {}))
            all_posts.extend(twitter_posts)
        
        # Add other social media platforms as needed
        
        logger.info(f"Collected {len(all_posts)} posts from social media")
        return all_posts
    
    def _collect_from_twitter(self, twitter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect tweets from Twitter.
        
        Args:
            twitter_config: Twitter-specific configuration
            
        Returns:
            List of tweets with metadata
        """
        logger.info("Collecting data from Twitter")
        
        # In a real implementation, you would use the Twitter API
        # This is a placeholder that would need to be implemented with proper API access
        
        hashtags = twitter_config.get('hashtags', [])
        accounts = twitter_config.get('accounts', [])
        max_tweets = twitter_config.get('max_tweets', 100)
        
        logger.warning("Twitter collection not fully implemented - requires API access")
        
        # Placeholder for demonstration
        tweets = []
        
        # In a real implementation, you would:
        # 1. Authenticate with Twitter API
        # 2. Search for tweets with the specified hashtags
        # 3. Get tweets from the specified accounts
        # 4. Process and return the results
        
        return tweets
    
    def _collect_from_industry_reports(self) -> List[Dict[str, Any]]:
        """
        Collect content from industry reports.
        
        Returns:
            List of industry reports with metadata
        """
        logger.info("Collecting data from industry reports")
        
        reports_config = self.sources.get('industry_reports', {})
        sources = reports_config.get('sources', [])
        max_reports = reports_config.get('max_reports', 5)
        
        all_reports = []
        
        for source in sources:
            try:
                name = source.get('name', 'Unknown')
                url = source.get('url', '')
                
                logger.info(f"Collecting from {name} ({url})")
                
                # In a real implementation, you would need site-specific scraping logic
                # This is a simplified placeholder
                
                reports = self._scrape_industry_reports(url, name, max_reports)
                all_reports.extend(reports)
                
                logger.info(f"Collected {len(reports)} reports from {name}")
                
            except Exception as e:
                logger.error(f"Error collecting data from {source.get('name', 'Unknown')}: {e}")
        
        logger.info(f"Collected {len(all_reports)} industry reports")
        return all_reports
    
    def _scrape_industry_reports(self, url: str, source_name: str, max_reports: int) -> List[Dict[str, Any]]:
        """
        Scrape industry reports from a website.
        
        Args:
            url: Website URL
            source_name: Name of the source
            max_reports: Maximum number of reports to scrape
            
        Returns:
            List of scraped industry reports
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a simplified implementation and would need to be customized for each site
            
            reports = []
            
            # Look for report elements (common in research pages)
            report_elements = soup.find_all(['div', 'article'], class_=['report', 'research', 'publication'])[:max_reports]
            
            if not report_elements:
                # Fallback to generic article elements
                report_elements = soup.find_all(['div', 'article'])[:max_reports]
            
            for element in report_elements:
                # Extract title
                title_elem = element.find(['h1', 'h2', 'h3', 'h4'])
                title = title_elem.text.strip() if title_elem else 'Unknown Title'
                
                # Extract link
                link_elem = title_elem.find('a') if title_elem else None
                if not link_elem:
                    link_elem = element.find('a')
                
                link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else ''
                
                # Handle relative URLs
                if link.startswith('/'):
                    link = f"{url.rstrip('/')}{link}"
                
                # Extract summary
                summary_elem = element.find(['p', 'div'], class_=['summary', 'excerpt', 'description'])
                summary = summary_elem.text.strip() if summary_elem else ''
                
                # Extract date
                date_elem = element.find(['time', 'span'], class_=['date', 'published', 'time'])
                published = date_elem.text.strip() if date_elem else ''
                
                report = {
                    'title': title,
                    'summary': summary,
                    'content': '',  # Would require visiting each report URL
                    'published': published,
                    'link': link,
                    'source': source_name,
                    'source_url': url,
                }
                
                reports.append(report)
            
            return reports
        
        except Exception as e:
            logger.error(f"Error scraping industry reports from {url}: {e}")
            return [] 