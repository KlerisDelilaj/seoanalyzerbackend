import pandas as pd
import numpy as np
from collections import Counter, defaultdict

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.clean_data()
    
    def clean_data(self):
        self.df['meta_description'] = self.df['meta_description'].fillna('')
        self.df['meta_description_length'] = self.df['meta_description_length'].fillna(0)
        
        numeric_cols = ['word_count', 'image_count', 'heading_count', 'internal_links', 
                        'external_links', 'meta_description_length', 'h1_count', 'h2_count', 
                        'h3_count', 'paragraph_count', 'avg_paragraph_length', 'page_size_kb']
        
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        self.df['contains_schema'] = self.df['contains_schema'].astype(bool)
        
        self.df['main_keyword'] = self.df.apply(self.get_main_keyword, axis=1)
        self.df['main_keyword_phrase'] = self.df.apply(self.get_main_phrase, axis=1)
        
        self.df['main_keyword_frequency'] = self.df.apply(
         lambda row: row['keywords'][0][1] if isinstance(row['keywords'], list) and len(row['keywords']) > 0 else 0, 
         axis=1
        )
        
        self.df['keyword_density'] = (self.df['main_keyword_frequency'] / self.df['word_count'] * 100).fillna(0)
        
        self.df['keyword_relevance'] = self.df.apply(
            lambda row: row['keywords'][0][2] if isinstance(row['keywords'], list) and len(row['keywords']) > 0 else 0,
            axis=1
        )
        
        unhashable_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                non_null_values = self.df[col].dropna()
                if len(non_null_values) > 0:
                    first_value = non_null_values.iloc[0]
                    if isinstance(first_value, (list, dict)):
                        unhashable_cols.append(col)
        
        hashable_cols = [col for col in self.df.columns if col not in unhashable_cols]
        if hashable_cols:
            duplicate_check_df = self.df[hashable_cols]
            if duplicate_check_df.duplicated().any():
                print(f"Found {duplicate_check_df.duplicated().sum()} duplicate rows. Removing...")
                duplicate_indices = duplicate_check_df[duplicate_check_df.duplicated()].index
                self.df = self.df.drop(index=duplicate_indices)
        
        q1 = self.df['word_count'].quantile(0.25)
        q3 = self.df['word_count'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = self.df[(self.df['word_count'] < lower_bound) | (self.df['word_count'] > upper_bound)]
        if not outliers.empty:
            print(f"Found {len(outliers)} outliers in word_count column")
            self.df['is_outlier'] = ((self.df['word_count'] < lower_bound) | 
                                     (self.df['word_count'] > upper_bound))
        else:
            self.df['is_outlier'] = False
    
    def get_main_keyword(self, row):
        if isinstance(row['keywords'], list) and len(row['keywords']) > 0:
            return row['keywords'][0][0]
        return ''
    
    def get_main_phrase(self, row):
        if isinstance(row['keyword_phrases'], list) and len(row['keyword_phrases']) > 0:
            return row['keyword_phrases'][0]['phrase']
        return ''
    
    def get_descriptive_stats(self):
        stats = {}
        
        numeric_cols = ['word_count', 'image_count', 'heading_count', 'internal_links', 
                        'external_links', 'meta_description_length', 'h1_count', 'h2_count', 
                        'h3_count', 'paragraph_count', 'avg_paragraph_length', 'page_size_kb',
                        'keyword_density', 'keyword_relevance']
        
        describe_dict = self.df[numeric_cols].describe().to_dict()
        for col in describe_dict:
            for stat in describe_dict[col]:
                if pd.isna(describe_dict[col][stat]):
                    describe_dict[col][stat] = None
        stats['numeric'] = describe_dict
        
        corr_dict = self.df[numeric_cols].corr().to_dict()
        for col in corr_dict:
            for stat in corr_dict[col]:
                if pd.isna(corr_dict[col][stat]):
                    corr_dict[col][stat] = None
        stats['correlations'] = corr_dict
        
        all_keywords = []
        for _, row in self.df.iterrows():
            if isinstance(row['keywords'], list):
                for kw, count, relevance in row['keywords']:
                    all_keywords.append({
                        'text': kw,
                        'count': count,
                        'relevance': relevance,
                        'is_phrase': False
                    })
            
            if isinstance(row['keyword_phrases'], list):
                for phrase_dict in row['keyword_phrases']:
                    all_keywords.append({
                        'text': phrase_dict['phrase'],
                        'count': phrase_dict['count'],
                        'relevance': phrase_dict['relevance'],
                        'is_phrase': True,
                        'word_count': phrase_dict['words']
                    })
        
        keyword_data = defaultdict(lambda: {'count': 0, 'relevance': 0, 'page_count': 0, 'is_phrase': False})
        
        for kw in all_keywords:
            key = kw['text']
            keyword_data[key]['count'] += kw['count']
            keyword_data[key]['relevance'] += kw['relevance']
            keyword_data[key]['page_count'] += 1
            keyword_data[key]['is_phrase'] = kw.get('is_phrase', False)
        
        keyword_list = [
            {
                'name': k, 
                'value': v['count'],
                'relevance': v['relevance'],
                'page_count': v['page_count'],
                'is_phrase': v['is_phrase']
            }
            for k, v in keyword_data.items()
        ]
        
        keyword_list.sort(key=lambda x: (x['relevance'], x['value']), reverse=True)
        
        stats['common_keywords'] = keyword_list[:20]
        
        phrases = [kw for kw in keyword_list if kw['is_phrase']]
        stats['common_phrases'] = phrases[:15]
        
        top_content = self.df.sort_values('word_count', ascending=False)[['url', 'title', 'word_count', 'content', 'page_link']].head(5)
        stats['top_content_pages'] = top_content.to_dict('records')
        
        top_images = self.df.sort_values('image_count', ascending=False)[['url', 'title', 'image_count', 'page_link']].head(5)
        stats['top_image_pages'] = top_images.to_dict('records')
        
        self.df['total_links'] = self.df['internal_links'] + self.df['external_links']
        top_links = self.df.sort_values('total_links', ascending=False)[['url', 'title', 'internal_links', 'external_links', 'total_links', 'page_link']].head(5)
        stats['top_link_pages'] = top_links.to_dict('records')
        
        keyword_by_page = []
        for _, row in self.df.iterrows():
            page_keywords = {
                'url': row['url'],
                'title': row['title'],
                'page_link': row['page_link'],
                'single_keywords': [],
                'phrases': []
            }
            
            if isinstance(row['keywords'], list):
                for kw, count, relevance in row['keywords'][:5]:
                    page_keywords['single_keywords'].append({
                        'keyword': kw,
                        'count': count,
                        'relevance': relevance
                    })
            
            if isinstance(row['keyword_phrases'], list):
                for phrase_dict in row['keyword_phrases'][:3]:
                    page_keywords['phrases'].append({
                        'phrase': phrase_dict['phrase'],
                        'count': phrase_dict['count'],
                        'relevance': phrase_dict['relevance']
                    })
            
            keyword_by_page.append(page_keywords)
        
        stats['keywords_by_page'] = keyword_by_page
        
        page_metrics = []
        for _, row in self.df.iterrows():
            metrics = {
                'url': row['url'].split('/')[-1] if '/' in row['url'] else row['url'],
                'wordCount': int(row['word_count']),
                'imageCount': int(row['image_count']),
                'internalLinks': int(row['internal_links']),
                'externalLinks': int(row['external_links']),
                'h1Count': int(row['h1_count']),
                'h2Count': int(row['h2_count']),
                'h3Count': int(row['h3_count']),
                'keywordDensity': float(row['keyword_density']),
                'mainKeyword': row['main_keyword'],
                'mainPhrase': row['main_keyword_phrase'],
                'pageLink': row['page_link']
            }
            page_metrics.append(metrics)
        
        stats['pageMetrics'] = page_metrics
        
        heading_distribution = [
            {"name": "H1", "value": int(self.df['h1_count'].sum())},
            {"name": "H2", "value": int(self.df['h2_count'].sum())},
            {"name": "H3", "value": int(self.df['h3_count'].sum())}
        ]
        stats['headingDistribution'] = heading_distribution
        
        link_distribution = [
            {"name": "Internal Links", "value": int(self.df['internal_links'].sum())},
            {"name": "External Links", "value": int(self.df['external_links'].sum())}
        ]
        stats['linkDistribution'] = link_distribution
        
        return stats
    
    def create_recommendations(self):
        recommendations = {
            'general': [],
            'content': [],
            'keywords': [],
            'structure': [],
            'pages_to_improve': []
        }
        
        avg_word_count = self.df['word_count'].mean()
        recommendations['general'].append({
            'text': f"Average word count across all pages is {avg_word_count:.1f}.",
            'recommendation': "Consider adding more content to your pages. The recommended minimum is around 300 words per page." if avg_word_count < 300 else "Your content length is good.",
            'type': 'warning' if avg_word_count < 300 else 'success'
        })
        
        avg_images = self.df['image_count'].mean()
        recommendations['general'].append({
            'text': f"Average number of images per page is {avg_images:.1f}.",
            'recommendation': "Consider adding more images to make your content more engaging." if avg_images < 2 else "Your image usage is good.",
            'type': 'warning' if avg_images < 2 else 'success'
        })
        
        if self.df['h1_count'].mean() != 1:
            recommendations['content'].append({
                'text': "H1 heading issues detected.",
                'recommendation': "Ensure each page has exactly one H1 heading. Some of your pages have missing or multiple H1 headings.",
                'type': 'warning'
            })
        
        if self.df['meta_description_length'].mean() < 120:
            recommendations['content'].append({
                'text': f"Average meta description length: {self.df['meta_description_length'].mean():.1f} characters.",
                'recommendation': "Your meta descriptions are too short on average. Aim for 150-160 characters to maximize visibility in search results.",
                'type': 'warning'
            })
        elif self.df['meta_description_length'].mean() > 160:
            recommendations['content'].append({
                'text': f"Average meta description length: {self.df['meta_description_length'].mean():.1f} characters.",
                'recommendation': "Your meta descriptions are too long on average. Keep them under 160 characters to prevent truncation in search results.",
                'type': 'warning'
            })
        
        if self.df['keyword_density'].mean() < 0.5:
            recommendations['keywords'].append({
                'text': f"Average keyword density: {self.df['keyword_density'].mean():.2f}%",
                'recommendation': "Your keyword density is low. Consider increasing the usage of your main keywords to 1-2% of your content.",
                'type': 'warning'
            })
        elif self.df['keyword_density'].mean() > 3:
            recommendations['keywords'].append({
                'text': f"Average keyword density: {self.df['keyword_density'].mean():.2f}%",
                'recommendation': "Your keyword density is too high, which might look like keyword stuffing to search engines. Aim for 1-2%.",
                'type': 'warning'
            })
        
        if not self.df['main_keyword_phrase'].str.strip().any():
            recommendations['keywords'].append({
                'text': "Limited keyword phrase usage detected.",
                'recommendation': "Consider using more 2-3 word keyword phrases throughout your content. Multi-word phrases often have less competition and can help target more specific search intent.",
                'type': 'warning'
            })
        
        if self.df['internal_links'].mean() < 3:
            recommendations['structure'].append({
                'text': f"Average internal links per page: {self.df['internal_links'].mean():.1f}",
                'recommendation': "Your pages have few internal links. Add more links between your pages to improve navigation and SEO.",
                'type': 'warning'
            })
        
        if not self.df['contains_schema'].any():
            recommendations['structure'].append({
                'text': "Schema markup is missing",
                'recommendation': "None of your pages use structured data (schema markup). Adding this can help search engines understand your content better.",
                'type': 'warning'
            })
        
        low_content_pages = self.df[self.df['word_count'] < 300][['url', 'title', 'word_count', 'page_link']]
        if not low_content_pages.empty:
            recommendations['pages_to_improve'].append({
                'issue': 'Low content',
                'description': 'These pages have less than 300 words and should be expanded:',
                'pages': low_content_pages.head(5).to_dict('records')
            })
        
        no_h1_pages = self.df[self.df['h1_count'] != 1][['url', 'title', 'h1_count', 'page_link']]
        if not no_h1_pages.empty:
            recommendations['pages_to_improve'].append({
                'issue': 'H1 heading issues',
                'description': 'These pages have missing or multiple H1 headings:',
                'pages': no_h1_pages.head(5).to_dict('records')
            })
        
        no_meta_desc = self.df[self.df['meta_description_length'] == 0][['url', 'title', 'page_link']]
        if not no_meta_desc.empty:
            recommendations['pages_to_improve'].append({
                'issue': 'Missing meta description',
                'description': 'These pages have no meta description:',
                'pages': no_meta_desc.head(5).to_dict('records')
            })
        
        low_keyword_relevance = self.df[self.df['keyword_relevance'] < 2][['url', 'title', 'main_keyword', 'main_keyword_phrase', 'page_link']]
        if not low_keyword_relevance.empty:
            recommendations['pages_to_improve'].append({
                'issue': 'Low keyword relevance',
                'description': 'These pages may need better keyword targeting:',
                'pages': low_keyword_relevance.head(5).to_dict('records')
            })
        
        return recommendations