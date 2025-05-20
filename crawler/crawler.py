import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import os
from collections import Counter, OrderedDict
import xml.etree.ElementTree as ET
import time
import random
from urllib.parse import urlparse, urljoin
import html

class WebsiteCrawler:
    def __init__(self, start_url, max_pages=20):
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited_urls = set()
        self.to_visit = [start_url]
        self.domain = urlparse(start_url).netloc
        self.data = []
        self.sitemap_urls = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_sitemap_urls(self):
        sitemap_url = urljoin(self.start_url, '/sitemap.xml')
        try:
            response = requests.get(sitemap_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)

                urls = []
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.append(url.text)
                
                sitemapindex_urls = []
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc = sitemap.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None and loc.text:
                        sitemapindex_urls.append(loc.text)
                
                for nested_sitemap_url in sitemapindex_urls:
                    try:
                        nested_response = requests.get(nested_sitemap_url, headers=self.headers, timeout=10)
                        if nested_response.status_code == 200:
                            nested_root = ET.fromstring(nested_response.content)
                            for url in nested_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                                urls.append(url.text)
                    except Exception as e:
                        print(f"Error accessing nested sitemap {nested_sitemap_url}: {e}")
                
                urls = [url for url in urls if self.domain in url]
                
                urls = [url for url in urls if not url.endswith('.xml')]
                
                self.sitemap_urls = urls
                return urls
            else:
                print(f"Sitemap not found at {sitemap_url}")
                return []
        except Exception as e:
            print(f"Error accessing sitemap: {e}")
            return []
    
    def extract_keywords(self, text, title='', meta_description='', num_keywords=10):
        import re
        from collections import Counter
        
        try:
            stopwords = ['the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'for', 'with', 'on', 'by', 
                        'this', 'that', 'be', 'are', 'as', 'i', 'you', 'he', 'she', 'we', 'they',
                        'was', 'were', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'may', 
                        'might', 'should', 'shall', 'must', 'do', 'does', 'did', 'but', 'or', 'if',
                        'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just', 'use',
                        'get', 'make', 'like', 'using', 'used', 'would', 'also', 'may', 'one', 'well',
                        'many', 'could', 'much', 'even', 'new', 'see', 'time', 'way']
            
            try:
                import nltk
                
                nltk_data_dir = os.path.expanduser("~/nltk_data")
                os.makedirs(nltk_data_dir, exist_ok=True)
                
                try:
                    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
                    nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
                    nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
                    
                    from nltk.corpus import stopwords as nltk_stopwords
                    from nltk.tokenize import word_tokenize
                    from nltk.stem import WordNetLemmatizer
                    
                    print("NLTK resources downloaded successfully")
                    
                    all_text = (title + " " + title + " " + meta_description + " " + 
                               meta_description + " " + text).lower()
                    
                    words = word_tokenize(all_text)
                    
                    stop_words = set(nltk_stopwords.words('english'))
                    stop_words.update(stopwords)
                    
                    lemmatizer = WordNetLemmatizer()
                    
                    processed_words = []
                    for word in words:
                        if (
                            word.isalpha() and
                            len(word) > 2 and
                            word not in stop_words
                        ):
                            lemma = lemmatizer.lemmatize(word)
                            processed_words.append(lemma)
                    
                    word_counts = Counter(processed_words)
                    
                    try:
                        bigrams = list(nltk.bigrams(processed_words))
                        trigrams = list(nltk.trigrams(processed_words))
                        
                        bigram_counts = Counter(bigrams)
                        trigram_counts = Counter(trigrams)
                        
                        use_advanced = True
                    except Exception as e:
                        print(f"NLTK n-gram extraction failed: {e}")
                        use_advanced = False
                    
                except Exception as e:
                    print(f"Error with NLTK processing: {e}")
                    use_advanced = False
                    
            except ImportError:
                print("NLTK not available, using basic extraction")
                use_advanced = False
            
            if 'use_advanced' not in locals() or not use_advanced:
                all_text = (title + " " + meta_description + " " + text).lower()
                words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text)
                words = [word for word in words if word not in stopwords]
                word_counts = Counter(words)
                bigram_counts = Counter()
                trigram_counts = Counter()
            
            keywords = []
            
            for word, count in word_counts.most_common(num_keywords):
                relevance = count * (0.5 + min(len(word) / 10.0, 0.5))
                keywords.append((word, count, relevance))
            
            if 'use_advanced' in locals() and use_advanced:
                for bigram, count in bigram_counts.most_common(num_keywords // 2):
                    if count > 1:
                        phrase = " ".join(bigram)
                        relevance = count * 1.2
                        keywords.append((phrase, count, relevance))
                
                for trigram, count in trigram_counts.most_common(num_keywords // 4):
                    if count > 1:
                        phrase = " ".join(trigram)
                        relevance = count * 1.5
                        keywords.append((phrase, count, relevance))
            
            title_words = set(re.findall(r'\b[a-zA-Z]{3,15}\b', title.lower()))
            meta_words = set(re.findall(r'\b[a-zA-Z]{3,15}\b', meta_description.lower()))
            
            for i, (keyword, count, relevance) in enumerate(keywords):
                keyword_words = set(keyword.split())
                
                if keyword_words.intersection(title_words):
                    keywords[i] = (keyword, count, relevance * 1.5)
                    
                if keyword_words.intersection(meta_words):
                    keywords[i] = (keyword, count, relevance * 1.3)
            
            keywords.sort(key=lambda x: x[2], reverse=True)
            
            return keywords[:num_keywords]
            
        except Exception as e:
            print(f"Keyword extraction failed with error: {e}")
            word_list = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
            word_list = [w for w in word_list if w not in stopwords]
            word_counts = Counter(word_list).most_common(num_keywords)
            return [(word, count, count) for word, count in word_counts]
    
    def find_content_area(self, soup):
        main_content_selectors = [
            'main', 'article', '#content', '.content', 
            '#main-content', '.main-content', '.post-content', '.entry-content',
            '.page-content', '#primary', '.site-content', '[role="main"]'
        ]
        
        content_area = None
        for selector in main_content_selectors:
            content_area = soup.select_one(selector)
            if content_area and len(content_area.get_text(strip=True)) > 100:
                return content_area
        
        body = soup.find('body')
        if not body:
            return soup
            
        content_area = BeautifulSoup(str(body), 'html.parser')
        
        elements_to_remove = [
            'header', '.header', '#header', 'nav', '.nav', '#nav',
            'footer', '.footer', '#footer', '.site-footer',
            '.sidebar', '#sidebar', 'aside', '.widget', '.widgets',
            '.advertisement', '.ads', '.ad-container',
            '.menu', '#menu', '.navigation', '.social-links',
            '.site-header', '.site-footer', '.comments', '#comments',
            '.cookie-notice', '.popup', '.modal'
        ]
        
        for selector in elements_to_remove:
            for element in content_area.select(selector):
                element.decompose()
        
        return content_area
    
    def extract_structured_content(self, soup):
        structured_content = {
            'html': str(soup),
            'headings': [],
            'sections': []
        }

        content_area = self.find_content_area(soup)
        
        headings = content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            structured_content['headings'].append({
                'level': int(heading.name[1]),
                'text': heading.get_text(strip=True),
                'html': str(heading)
            })
        
        current_heading = None
        current_content = []
        processed_elements = set()
        
        top_level_elements = []
        for element in content_area.children:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'div']:
                top_level_elements.append(element)
        
        for element in top_level_elements:
            if element in processed_elements:
                continue
                
            processed_elements.add(element)
            
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_heading and current_content:
                    structured_content['sections'].append({
                        'heading': current_heading,
                        'content': current_content
                    })
                
                current_heading = {
                    'level': int(element.name[1]),
                    'text': element.get_text(strip=True),
                    'html': str(element)
                }
                current_content = []
            elif element.name in ['p', 'ul', 'ol', 'table']:
                content_text = element.get_text(strip=True)
                if content_text:
                    current_content.append({
                        'type': element.name,
                        'text': content_text,
                        'html': str(element)
                    })
            elif element.name == 'div':
                has_heading = bool(element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
                
                if not has_heading:
                    inner_elements = element.find_all(['p', 'ul', 'ol', 'table'])
                    
                    if inner_elements:
                        for inner in inner_elements:
                            processed_elements.add(inner)
                            content_text = inner.get_text(strip=True)
                            if content_text:
                                current_content.append({
                                    'type': inner.name,
                                    'text': content_text,
                                    'html': str(inner)
                                })
                    else:
                        content_text = element.get_text(strip=True)
                        if content_text:
                            current_content.append({
                                'type': 'div',
                                'text': content_text,
                                'html': str(element)
                            })
        
        if current_heading and current_content:
            structured_content['sections'].append({
                'heading': current_heading,
                'content': current_content
            })
        
        if not structured_content['sections'] and len(content_area.get_text(strip=True)) > 0:
            paragraphs = []
            for p in content_area.find_all(['p', 'ul', 'ol', 'table']):
                if p not in processed_elements:
                    paragraphs.append(p)
            
            content = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    content.append({
                        'type': p.name,
                        'text': text,
                        'html': str(p)
                    })
            
            if content:
                structured_content['sections'].append({
                    'heading': None,
                    'content': content
                })
        
        return structured_content
    
    def analyze_page(self, url, soup):
        metrics = {
            'url': url,
            'title': '',
            'word_count': 0,
            'image_count': 0,
            'heading_count': 0,
            'internal_links': 0,
            'external_links': 0,
            'keywords': [],
            'keyword_phrases': [],
            'meta_description': '',
            'meta_description_length': 0,
            'h1_count': 0,
            'h2_count': 0,
            'h3_count': 0,
            'paragraph_count': 0,
            'avg_paragraph_length': 0,
            'contains_schema': False,
            'page_size_kb': 0,
            'content': '',
            'structured_content': {},
            'page_link': url,
            'raw_html': str(soup),
        }
        
        title_tag = soup.find('title')
        title_text = title_tag.text.strip() if title_tag else 'No Title'
        metrics['title'] = title_text
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description_text = ''
        if meta_desc and 'content' in meta_desc.attrs:
            meta_description_text = meta_desc['content']
            metrics['meta_description'] = meta_description_text
            metrics['meta_description_length'] = len(meta_description_text)
        
        content_area = self.find_content_area(soup)
        
        metrics['structured_content'] = self.extract_structured_content(content_area)
        
        text_content = ' '.join([p.text for p in content_area.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])])
        metrics['word_count'] = len(re.findall(r'\b\w+\b', text_content))
        
        metrics['content'] = text_content[:2000] + '...' if len(text_content) > 2000 else text_content
        
        keywords = self.extract_keywords(text_content, title_text, meta_description_text, num_keywords=15)
        
        single_words = []
        phrases = []
        
        for kw, count, relevance in keywords:
            if ' ' in kw:
                phrases.append({
                    'phrase': kw,
                    'count': count,
                    'relevance': relevance,
                    'words': len(kw.split())
                })
            else:
                single_words.append((kw, count, relevance))
        
        metrics['keywords'] = single_words
        metrics['keyword_phrases'] = phrases
        
        metrics['image_count'] = len(content_area.find_all('img'))
        
        headings = content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        metrics['heading_count'] = len(headings)
        metrics['h1_count'] = len(content_area.find_all('h1'))
        metrics['h2_count'] = len(content_area.find_all('h2'))
        metrics['h3_count'] = len(content_area.find_all('h3'))
        
        paragraphs = content_area.find_all('p')
        metrics['paragraph_count'] = len(paragraphs)
        if paragraphs:
            total_length = sum(len(p.text.split()) for p in paragraphs)
            metrics['avg_paragraph_length'] = total_length / len(paragraphs)
        
        all_links = content_area.find_all('a', href=True)
        internal_links = 0
        external_links = 0
        
        for link in all_links:
            href = link['href']
            if href.startswith('#') or not href:
                continue
            elif href.startswith('/') or self.domain in href:
                internal_links += 1
            else:
                external_links += 1
        
        metrics['internal_links'] = internal_links
        metrics['external_links'] = external_links
        
        metrics['contains_schema'] = bool(soup.find_all('script', type='application/ld+json'))
        
        metrics['page_size_kb'] = len(str(soup)) / 1024
        
        return metrics
    
    def crawl_single_page(self, url=None):
        if url is None:
            url = self.start_url
            
        print(f"Analyzing single page: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                metrics = self.analyze_page(url, soup)
                self.data = [metrics]
                self.visited_urls.add(url)
                print(f"Successfully analyzed page: {url}")
                
                self.get_sitemap_urls()
                
                return pd.DataFrame(self.data)
            else:
                print(f"Failed to access page: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error analyzing page {url}: {e}")
            return pd.DataFrame()
    
    def crawl(self):
        df = self.crawl_single_page()
        
        if df.empty:
            return df
        
        if not self.sitemap_urls:
            self.get_sitemap_urls()
            
        if self.sitemap_urls:
            print(f"Found {len(self.sitemap_urls)} URLs in sitemap")
            for url in self.sitemap_urls:
                if url not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                    self.to_visit.append(url)
        else:
            response = requests.get(self.start_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for link in links:
                    url = link['href']
                    if url.startswith('/'):
                        url = urljoin(self.start_url, url)
                    
                    if self.domain in url and url not in self.visited_urls and url not in self.to_visit:
                        self.to_visit.append(url)
        
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            url = self.to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            try:
                time.sleep(random.uniform(0.5, 1.5))
                
                print(f"Crawling: {url}")
                
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    metrics = self.analyze_page(url, soup)
                    self.data.append(metrics)
                    self.visited_urls.add(url)
                    
                    links = soup.find_all('a', href=True)
                    for link in links:
                        new_url = link['href']
                        if new_url.startswith('/'):
                            new_url = urljoin(self.start_url, new_url)
                        
                        if self.domain in new_url and new_url not in self.visited_urls and new_url not in self.to_visit:
                            self.to_visit.append(new_url)
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        print(f"Crawling complete. Visited {len(self.visited_urls)} pages.")
        
        return pd.DataFrame(self.data)