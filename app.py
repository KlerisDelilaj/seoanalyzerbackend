from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from crawler.crawler import WebsiteCrawler
from crawler.analyzer import DataAnalyzer

os.environ['NLTK_DATA'] = os.path.join(os.path.expanduser('~'), 'nltk_data')

try:
    import nltk
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
    nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")
    print("The application will use a fallback keyword extraction method")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
CORS(app)

crawler_instance = None

@app.route('/api/analyze', methods=['POST'])
def analyze_website():
    data = request.json
    url = data.get('url')
    single_page = data.get('single_page', True)
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        global crawler_instance
        crawler_instance = WebsiteCrawler(url, max_pages=1 if single_page else 20)
        
        df = crawler_instance.crawl_single_page()
        
        if df.empty:
            return jsonify({'error': 'Failed to collect data from website'}), 400
        
        analyzer = DataAnalyzer(df)
        stats = analyzer.get_descriptive_stats()
        recommendations = analyzer.create_recommendations()
        
        page_data = df.to_dict('records')
        
        for page in page_data:
            if isinstance(page.get('keywords'), list):
                formatted_keywords = []
                for item in page['keywords']:
                    if isinstance(item, tuple) and len(item) == 2:
                        formatted_keywords.append({"name": item[0], "value": item[1]})
                page['keywords'] = formatted_keywords
        
        sitemap_urls = crawler_instance.sitemap_urls
        
        return jsonify({
            'status': 'success',
            'analyzed_url': url,
            'single_page_analysis': True,
            'pages': page_data,
            'stats': stats,
            'recommendations': recommendations,
            'page_count': len(df),
            'crawl_domain': crawler_instance.domain,
            'sitemap_urls': sitemap_urls
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sitemap', methods=['GET'])
def get_sitemap():
    global crawler_instance
    
    if not crawler_instance:
        return jsonify({'error': 'No website has been analyzed yet'}), 400
    
    sitemap_urls = crawler_instance.sitemap_urls
    
    return jsonify({
        'status': 'success',
        'sitemap_urls': sitemap_urls,
        'count': len(sitemap_urls)
    })

@app.route('/api/analyze-url', methods=['POST'])
def analyze_specific_url():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        global crawler_instance
        if not crawler_instance:
            return jsonify({'error': 'No crawler instance available. Analyze a website first.'}), 400
        
        crawler = WebsiteCrawler(url, max_pages=1)
        df = crawler.crawl_single_page()
        
        if df.empty:
            return jsonify({'error': 'Failed to collect data from the specified URL'}), 400
        
        analyzer = DataAnalyzer(df)
        stats = analyzer.get_descriptive_stats()
        recommendations = analyzer.create_recommendations()
        
        page_data = df.to_dict('records')
        
        for page in page_data:
            if isinstance(page.get('keywords'), list):
                formatted_keywords = []
                for item in page['keywords']:
                    if isinstance(item, tuple) and len(item) == 2:
                        formatted_keywords.append({"name": item[0], "value": item[1]})
                page['keywords'] = formatted_keywords
        
        return jsonify({
            'status': 'success',
            'analyzed_url': url,
            'single_page_analysis': True,
            'pages': page_data,
            'stats': stats,
            'recommendations': recommendations,
            'page_count': 1,
            'crawl_domain': crawler.domain,
            'sitemap_urls': crawler_instance.sitemap_urls
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)