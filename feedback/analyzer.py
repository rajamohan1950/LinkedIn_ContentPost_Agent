"""
Feedback Analysis Module

This module is responsible for analyzing feedback and engagement data from LinkedIn posts.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """Analyzes feedback and engagement data from LinkedIn posts."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feedback analyzer.
        
        Args:
            config: Configuration dictionary for feedback analysis
        """
        self.config = config
        self.metrics_config = config.get('metrics', {}).get('engagement', {})
        self.learning_config = config.get('learning', {})
        self.ab_testing_config = config.get('a_b_testing', {})
        
        logger.info("Feedback analyzer initialized")
    
    def analyze(self, post_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze post results and generate insights.
        
        Args:
            post_results: List of post results with engagement data
            
        Returns:
            Dictionary containing analysis results
        """
        if not post_results:
            logger.warning("No post results to analyze")
            return {
                'engagement_score': 0,
                'performance': 'no_data',
                'recommendations': []
            }
        
        # Convert post results to engagement data format
        engagement_data = self._convert_to_engagement_data(post_results)
        
        # Analyze different aspects
        content_type_performance = self.analyze_content_type_performance(engagement_data)
        topic_performance = self.analyze_topic_performance(engagement_data)
        timing_performance = self.analyze_timing_performance(engagement_data)
        trends = self.analyze_engagement_trends(engagement_data)
        
        # Generate overall engagement score
        engagement_score = statistics.mean([
            self.calculate_engagement_score(post['engagement'])
            for post in engagement_data['posts']
        ]) if engagement_data['posts'] else 0
        
        # Generate recommendations
        recommendations = self.generate_recommendations(engagement_data)
        
        return {
            'engagement_score': engagement_score,
            'performance': self._categorize_performance(engagement_score),
            'content_type_performance': content_type_performance,
            'topic_performance': topic_performance,
            'timing_performance': timing_performance,
            'trends': trends,
            'recommendations': recommendations
        }
    
    def analyze_post_performance(self, post_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance of a single post.
        
        Args:
            post_result: Post result with engagement data
            
        Returns:
            Dictionary containing performance analysis
        """
        if not post_result.get('success', False):
            logger.warning("Invalid post results")
            return {'engagement_score': 0, 'performance_category': 'failed'}
        
        engagement = post_result.get('engagement', {})
        engagement_score = self.calculate_engagement_score(engagement)
        
        return {
            'engagement_score': engagement_score,
            'performance_category': self._categorize_performance(engagement_score),
            'metrics': engagement
        }
    
    def analyze_content_type_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance by content type.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            Dictionary containing content type performance analysis
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {}
        
        # Group posts by content type
        content_types = {}
        for post in engagement_data['posts']:
            content_type = post.get('content_type')
            if content_type:
                if content_type not in content_types:
                    content_types[content_type] = []
                content_types[content_type].append(post)
        
        # Calculate performance for each content type
        performance = {}
        for content_type, posts in content_types.items():
            scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
            avg_score = statistics.mean(scores) if scores else 0
            trend = self._calculate_trend(scores)
            
            performance[content_type] = {
                'score': avg_score,
                'trend': trend,
                'posts': len(posts)
            }
        
        return performance
    
    def analyze_topic_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance by topic.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            Dictionary containing topic performance analysis
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {}
        
        # Group posts by topic
        topics = {}
        for post in engagement_data['posts']:
            for topic in post.get('topics', []):
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(post)
        
        # Calculate performance for each topic
        performance = {}
        for topic, posts in topics.items():
            scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
            avg_score = statistics.mean(scores) if scores else 0
            trend = self._calculate_trend(scores)
            
            performance[topic] = {
                'engagement_score': avg_score,
                'trend': trend,
                'posts': len(posts)
            }
        
        return performance
    
    def analyze_timing_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance by posting time.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            Dictionary containing timing performance analysis
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {'best_days': [], 'best_hours': []}
        
        # Group posts by day and hour
        days = {}
        hours = {}
        
        for post in engagement_data['posts']:
            # Get day and hour from posted_at
            posted_at = datetime.fromisoformat(post['posted_at'])
            day = posted_at.strftime('%A').lower()
            hour = posted_at.strftime('%H:00')
            
            # Add to days
            if day not in days:
                days[day] = []
            days[day].append(post)
            
            # Add to hours
            if hour not in hours:
                hours[hour] = []
            hours[hour].append(post)
        
        # Calculate best days
        day_performance = {}
        for day, posts in days.items():
            scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
            avg_score = statistics.mean(scores) if scores else 0
            day_performance[day] = avg_score
        
        # Calculate best hours
        hour_performance = {}
        for hour, posts in hours.items():
            scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
            avg_score = statistics.mean(scores) if scores else 0
            hour_performance[hour] = avg_score
        
        # Sort by performance
        best_days = sorted(day_performance.items(), key=lambda x: x[1], reverse=True)
        best_hours = sorted(hour_performance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'best_days': [{'day': day, 'score': score} for day, score in best_days],
            'best_hours': [{'hour': hour, 'score': score} for hour, score in best_hours]
        }
    
    def analyze_engagement_trends(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze engagement trends over time.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            Dictionary containing trend analysis
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {'overall_trend': 'stable', 'metrics_trends': {}, 'topics_trends': {}}
        
        # Sort posts by date
        sorted_posts = sorted(engagement_data['posts'],
                            key=lambda x: datetime.fromisoformat(x['posted_at']))
        
        # Calculate overall trend
        scores = [self.calculate_engagement_score(post['engagement']) for post in sorted_posts]
        overall_trend = self._calculate_trend(scores)
        
        # Calculate trends for each metric
        metrics_trends = {}
        for metric in ['impressions', 'likes', 'comments', 'shares']:
            values = [post['engagement'].get(metric, 0) for post in sorted_posts]
            metrics_trends[metric] = self._calculate_trend(values)
        
        # Calculate trends for each topic
        topics_trends = {}
        for post in sorted_posts:
            for topic in post.get('topics', []):
                if topic not in topics_trends:
                    topics_trends[topic] = []
                topics_trends[topic].append(self.calculate_engagement_score(post['engagement']))
        
        topics_trends = {
            topic: self._calculate_trend(scores)
            for topic, scores in topics_trends.items()
        }
        
        return {
            'overall_trend': overall_trend,
            'metrics_trends': metrics_trends,
            'topics_trends': topics_trends
        }
    
    def analyze_ab_test_results(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Args:
            test_data: Dictionary containing A/B test data
            
        Returns:
            Dictionary containing A/B test analysis
        """
        if not test_data.get('posts'):
            logger.warning("Insufficient data")
            return {}
        
        # Group posts by variant
        variants = {}
        for post in test_data['posts']:
            variant = post.get('variant')
            if variant:
                if variant not in variants:
                    variants[variant] = []
                variants[variant].append(post)
        
        # Need at least two variants with data
        if len(variants) < 2:
            logger.warning("Insufficient data")
            return {}
        
        # Calculate performance for each variant
        variant_performance = {}
        for variant, posts in variants.items():
            scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
            avg_score = statistics.mean(scores) if scores else 0
            variant_performance[variant] = {
                'score': avg_score,
                'sample_size': len(posts)
            }
        
        # Find winning variant
        winning_variant = max(variant_performance.items(), key=lambda x: x[1]['score'])
        baseline_variant = min(variant_performance.items(), key=lambda x: x[1]['score'])
        
        # Calculate improvement
        improvement = ((winning_variant[1]['score'] - baseline_variant[1]['score']) /
                      baseline_variant[1]['score'] * 100 if baseline_variant[1]['score'] > 0 else 0)
        
        # Calculate confidence level (simplified)
        confidence_level = min(improvement / 10, 0.99) if improvement > 0 else 0
        
        return {
            'winning_variant': winning_variant[0],
            'improvement': improvement,
            'confidence_level': confidence_level,
            'variant_performance': variant_performance
        }
    
    def generate_recommendations(self, engagement_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            List of recommendations
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return []
        
        recommendations = []
        
        # Analyze content type performance
        content_type_perf = self.analyze_content_type_performance(engagement_data)
        best_content_type = max(content_type_perf.items(), key=lambda x: x[1]['score'])[0]
        recommendations.append(f"Focus on creating more {best_content_type} content")
        
        # Analyze topic performance
        topic_perf = self.analyze_topic_performance(engagement_data)
        best_topic = max(topic_perf.items(), key=lambda x: x[1]['engagement_score'])[0]
        recommendations.append(f"Continue focusing on {best_topic} topics")
        
        # Analyze timing performance
        timing_perf = self.analyze_timing_performance(engagement_data)
        if timing_perf['best_days']:
            best_day = timing_perf['best_days'][0]['day']
            recommendations.append(f"Prioritize posting on {best_day}")
        
        if timing_perf['best_hours']:
            best_hour = timing_perf['best_hours'][0]['hour']
            recommendations.append(f"Target posting at {best_hour}")
        
        # Analyze trends
        trends = self.analyze_engagement_trends(engagement_data)
        if trends['overall_trend'] == 'decreasing':
            recommendations.append("Review and adjust content strategy to improve engagement")
        
        return recommendations
    
    def detect_anomalies(self, engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in engagement data.
        
        Args:
            engagement_data: Engagement data dictionary
            
        Returns:
            List of detected anomalies
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return []
        
        anomalies = []
        
        # Calculate baseline metrics
        all_scores = [self.calculate_engagement_score(post['engagement'])
                     for post in engagement_data['posts']]
        
        mean_score = statistics.mean(all_scores)
        stdev_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        
        # Check each post for anomalies
        for post in engagement_data['posts']:
            score = self.calculate_engagement_score(post['engagement'])
            
            # Check for unusually high or low engagement
            if abs(score - mean_score) > 2 * stdev_score:
                anomalies.append({
                    'post_id': post['post_id'],
                    'score': score,
                    'reason': 'unusual_engagement',
                    'details': {
                        'difference_from_mean': score - mean_score,
                        'standard_deviations': (score - mean_score) / stdev_score if stdev_score > 0 else 0
                    }
                })
        
        return anomalies
    
    def calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate engagement score based on metrics.
        
        Args:
            metrics: Dictionary containing engagement metrics
            
        Returns:
            Engagement score
        """
        if not metrics:
            return 0.0
        
        # Get weights from config or use defaults
        weights = {
            'likes': self.metrics_config.get('likes_weight', 1.0),
            'comments': self.metrics_config.get('comments_weight', 2.0),
            'shares': self.metrics_config.get('shares_weight', 3.0),
            'views': self.metrics_config.get('views_weight', 0.1)
        }
        
        # Calculate weighted sum
        score = sum(
            metrics.get(metric, 0) * weight
            for metric, weight in weights.items()
        )
        
        return score
    
    def save_analysis(self, analysis: Dict[str, Any], filepath: str) -> bool:
        """
        Save analysis results to a file.
        
        Args:
            analysis: Analysis results dictionary
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return False
    
    def load_analysis(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Analysis results dictionary or None if error
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis: {str(e)}")
            return None
    
    def _convert_to_engagement_data(self, post_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert post results to engagement data format.
        
        Args:
            post_results: List of post results
            
        Returns:
            Dictionary containing engagement data
        """
        posts = []
        for post in post_results:
            if post.get('success', False):
                content = post.get('content', {})
                posts.append({
                    'post_id': post['post_id'],
                    'content_type': content.get('content_type'),
                    'topics': content.get('topics', []),
                    'posted_at': post['posted_at'],
                    'engagement': post['engagement']
                })
        
        return {'posts': posts}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Trend direction ('increasing', 'decreasing', or 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate average change
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        avg_change = statistics.mean(changes) if changes else 0
        
        # Determine trend direction
        threshold = 0.1  # Minimum change to consider a trend
        if avg_change > threshold:
            return 'increasing'
        elif avg_change < -threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def _categorize_performance(self, score: float) -> str:
        """
        Categorize performance based on engagement score.
        
        Args:
            score: Engagement score
            
        Returns:
            Performance category
        """
        if score == 0:
            return 'no_engagement'
        elif score < 100:
            return 'poor'
        elif score < 200:
            return 'fair'
        elif score < 300:
            return 'good'
        else:
            return 'excellent' 