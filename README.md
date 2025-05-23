# SEO Analyzer

A full-stack web application for analyzing websites and providing SEO recommendations.

## Project Overview

The SEO Analyzer is a comprehensive tool that:

- Crawls websites to collect SEO-relevant data
- Analyzes content quality, keywords, and link structure
- Provides actionable recommendations for SEO improvement
- Visualizes data through interactive charts and reports

## Architecture

The application consists of:

1. **React Frontend**: User interface built with TypeScript and ReactJS
2. **Flask Backend API**: Handles requests and orchestrates analysis
3. **Web Crawler Module**: Collects data from websites
4. **Data Analyzer Module**: Processes data and generates recommendations

## Key Features

- Single page or multi-page website analysis
- Content quality assessment (word count, headings, etc.)
- Keyword extraction and relevance scoring
- Internal and external link analysis
- SEO recommendation engine
- Data visualization with charts
- Sitemap integration

## Technical Components

### Frontend (React)

Located in `paste.txt`, the frontend provides:

- User interface for entering URLs and viewing analysis
- Dashboard with overview statistics and charts
- Detailed views for content, links, keywords, and recommendations
- Interactive data visualizations using Recharts
- Page content viewer with structured and raw HTML options

### Backend API (Flask)

Located in `app.py`, the backend API provides:

- RESTful endpoints for website analysis
- Integration with crawler and analyzer modules
- JSON response formatting with custom encoder
- Error handling and validation
- CORS support for cross-origin requests

### Web Crawler Module

Located in `crawler.py`, the crawler module:

- Visits and parses web pages using BeautifulSoup
- Extracts page content, links, and metadata
- Uses NLP techniques to identify keywords and phrases
- Fetches and parses XML sitemaps
- Collects comprehensive metrics for each page

### Data Analyzer Module

Located in `analyzer.py`, the analyzer module:

- Cleans and preprocesses crawled data
- Calculates descriptive statistics and correlations
- Identifies main keywords and their relevance
- Generates actionable SEO recommendations
- Identifies pages that need improvement

## Data Flow

1. User enters a URL in the frontend
2. Frontend calls API function to request analysis
3. Backend initiates crawling process
4. Crawler visits pages and extracts data
5. Backend passes collected data to analyzer
6. Analyzer processes data and generates recommendations
7. Backend returns structured results to frontend
8. Frontend displays results in various visual formats

## API Endpoints

- `POST /api/analyze`: Analyzes a website by URL
- `GET /api/sitemap`: Returns sitemap URLs for a previously analyzed site
- `POST /api/analyze-url`: Analyzes a specific URL
- `GET /api/status`: Simple health check

## Installation and Setup

### Requirements

See `requirements.txt` for Python dependencies:

```
Flask==2.3.3
Flask-CORS==3.0.10
Werkzeug==2.3.7
requests==2.26.0
beautifulsoup4==4.10.0
pandas==1.5.3
numpy==1.24.3
nltk==3.8.1
gunicorn==20.1.0
```

### Backend Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. The API will be available at http://localhost:5000

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

3. The application will be available at http://localhost:3000

## Usage

1. Enter a URL in the input field
2. Click "Analyze Single Page" to analyze just the entered URL
3. Navigate between tabs to explore different aspects of the analysis
4. View recommendations for improving SEO
5. Explore sitemap URLs for more specific analysis

## Implementation Details

### WebsiteCrawler Class Methods

- `__init__(start_url, max_pages=20)`: Initialize crawler with URL and limit
- `get_sitemap_urls()`: Extract URLs from sitemap.xml
- `extract_keywords(text, title, meta_description)`: Extract and score keywords
- `find_content_area(soup)`: Identify the main content area of a page
- `extract_structured_content(soup)`: Create structured representation of content
- `analyze_page(url, soup)`: Collect comprehensive page metrics
- `crawl_single_page(url=None)`: Analyze a single URL
- `crawl()`: Analyze multiple pages up to max_pages limit

### DataAnalyzer Class Methods

- `__init__(df)`: Initialize analyzer with DataFrame of crawled data
- `clean_data()`: Preprocess and normalize the data
- `get_main_keyword(row)`: Extract primary keyword for a page
- `get_main_phrase(row)`: Extract primary keyword phrase
- `get_descriptive_stats()`: Calculate comprehensive statistics
- `create_recommendations()`: Generate actionable SEO advice

## Future Enhancements

- Authentication for accessing historical analyses
- Competitor analysis and comparison
- Backlink analysis and suggestions
- Page speed and mobile-friendliness assessment
- Integration with Google Search Console and Analytics
- Scheduled monitoring and alerts