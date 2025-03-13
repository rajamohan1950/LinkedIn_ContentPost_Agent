"""Main script to run the LinkedIn Content Post Agent."""
import os
import logging
from dotenv import load_dotenv
from src.feedback.analyzer import FeedbackAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the LinkedIn Content Post Agent."""
    # Load environment variables
    load_dotenv()
    
    # Initialize feedback analyzer with configuration
    config = {
        'min_posts': 2,
        'anomaly_threshold': 2.0,
        'metrics': {
            'engagement': {
                'likes_weight': 1.0,
                'comments_weight': 2.0,
                'shares_weight': 3.0,
                'views_weight': 0.1
            }
        }
    }
    
    analyzer = FeedbackAnalyzer(config)
    
    # Example post results
    post_results = [
        {
            'post_id': '1',
            'content': {
                'title': 'Understanding Deep Learning Fundamentals',
                'content_type': 'article',
                'topics': ['AI', 'Deep Learning']
            },
            'posted_at': '2025-03-13T10:00:00Z',
            'engagement': {
                'likes': 100,
                'comments': 20,
                'shares': 10,
                'views': 1000
            },
            'success': True
        },
        {
            'post_id': '2',
            'content': {
                'title': 'Machine Learning Best Practices',
                'content_type': 'article',
                'topics': ['Machine Learning']
            },
            'posted_at': '2025-03-12T15:00:00Z',
            'engagement': {
                'likes': 80,
                'comments': 15,
                'shares': 8,
                'views': 800
            },
            'success': True
        }
    ]
    
    try:
        # Analyze post results
        analysis = analyzer.analyze(post_results)
        
        # Log analysis results
        logger.info("Analysis Results:")
        logger.info(f"Overall Engagement Score: {analysis['engagement_score']:.2f}")
        logger.info(f"Performance Category: {analysis['performance']}")
        
        logger.info("\nContent Type Performance:")
        for content_type, perf in analysis['content_type_performance'].items():
            logger.info(f"- {content_type}: {perf['score']:.2f} (trend: {perf['trend']})")
        
        logger.info("\nTopic Performance:")
        for topic, perf in analysis['topic_performance'].items():
            logger.info(f"- {topic}: {perf['engagement_score']:.2f} (trend: {perf['trend']})")
        
        logger.info("\nTiming Performance:")
        logger.info("Best Days:")
        for day in analysis['timing_performance']['best_days']:
            logger.info(f"- {day}")
        
        logger.info("\nTrends:")
        logger.info(f"Overall Trend: {analysis['trends'].get('overall_trend', 'stable')}")
        
        # Save analysis results to file
        analyzer.save_analysis(analysis, 'analysis_results.json')
        logger.info("\nAnalysis results saved to analysis_results.json")
        
    except Exception as e:
        logger.error(f"Error running LinkedIn Content Post Agent: {str(e)}")
        raise

if __name__ == "__main__":
    main() 