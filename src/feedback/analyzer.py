"""Feedback analyzer module for analyzing LinkedIn post performance."""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """Analyzes feedback and engagement data from LinkedIn posts."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feedback analyzer.

        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config
        self.min_posts = config['analysis']['min_posts']
        self.anomaly_threshold = config['analysis']['anomaly_threshold']
        self.trend_window_days = config['analysis']['trend_window_days']
        self.engagement_weights = config['metrics']['engagement']
        logger.info("Feedback analyzer initialized")

    def calculate_engagement_score(self, engagement: Dict[str, int]) -> float:
        """
        Calculate engagement score based on configured weights.

        Args:
            engagement: Dictionary containing engagement metrics

        Returns:
            Weighted engagement score
        """
        weighted_sum = (
            engagement['likes'] * self.engagement_weights['likes_weight'] +
            engagement['comments'] * self.engagement_weights['comments_weight'] +
            engagement['shares'] * self.engagement_weights['shares_weight']
        )
        return weighted_sum / engagement['impressions'] if engagement['impressions'] > 0 else 0.0

    def analyze_content_type_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Analyze performance by content type.

        Args:
            engagement_data: Engagement data dictionary

        Returns:
            Dictionary containing best performing content types
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {'best_types': []}

        type_performance = {}
        for post in engagement_data['posts']:
            content_type = post.get('content_type', 'unknown')
            score = self.calculate_engagement_score(post['engagement'])
            if content_type not in type_performance:
                type_performance[content_type] = {'total_score': 0, 'count': 0}
            type_performance[content_type]['total_score'] += score
            type_performance[content_type]['count'] += 1

        # Calculate average scores and sort
        avg_scores = {
            ctype: data['total_score'] / data['count']
            for ctype, data in type_performance.items()
        }
        best_types = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return {'best_types': [t[0] for t in best_types[:3]]}

    def analyze_topic_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Analyze performance by topic.

        Args:
            engagement_data: Engagement data dictionary

        Returns:
            Dictionary containing best performing topics
        """
        if not engagement_data.get('posts'):
            logger.warning("No posts found")
            return {'best_topics': []}

        topic_performance = {}
        for post in engagement_data['posts']:
            topics = post.get('topics', ['unknown'])
            score = self.calculate_engagement_score(post['engagement'])
            for topic in topics:
                if topic not in topic_performance:
                    topic_performance[topic] = {'total_score': 0, 'count': 0}
                topic_performance[topic]['total_score'] += score
                topic_performance[topic]['count'] += 1

        # Calculate average scores and sort
        avg_scores = {
            topic: data['total_score'] / data['count']
            for topic, data in topic_performance.items()
        }
        best_topics = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return {'best_topics': [t[0] for t in best_topics[:3]]}

    def analyze_timing_performance(self, engagement_data: Dict[str, Any]) -> Dict[str, List[str]]:
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
            posted_at = datetime.fromisoformat(post['posted_at'])
            day = posted_at.strftime('%A').lower()
            hour = posted_at.hour

            score = self.calculate_engagement_score(post['engagement'])

            if day not in days:
                days[day] = {'total_score': 0, 'count': 0}
            days[day]['total_score'] += score
            days[day]['count'] += 1

            if hour not in hours:
                hours[hour] = {'total_score': 0, 'count': 0}
            hours[hour]['total_score'] += score
            hours[hour]['count'] += 1

        # Calculate average scores
        day_scores = {
            day: data['total_score'] / data['count']
            for day, data in days.items()
        }
        hour_scores = {
            hour: data['total_score'] / data['count']
            for hour, data in hours.items()
        }

        # Sort and get best performing times
        best_days = sorted(day_scores.items(), key=lambda x: x[1], reverse=True)
        best_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'best_days': [d[0] for d in best_days[:3]],
            'best_hours': [h[0] for h in best_hours[:3]]
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
            return {'trend_direction': 'stable', 'trend_strength': 0.0}

        # Sort posts by date
        sorted_posts = sorted(engagement_data['posts'],
                            key=lambda x: datetime.fromisoformat(x['posted_at']))

        # Calculate engagement scores over time
        scores = [self.calculate_engagement_score(post['engagement'])
                 for post in sorted_posts]

        if len(scores) < 2:
            return {'trend_direction': 'stable', 'trend_strength': 0.0}

        # Simple linear trend analysis
        trend = sum(y - x for x, y in zip(scores[:-1], scores[1:])) / (len(scores) - 1)
        trend_strength = abs(trend)
        trend_direction = 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'

        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }

    def analyze_post_performance(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze individual post performance.

        Args:
            post: Post data dictionary

        Returns:
            Dictionary containing post performance analysis
        """
        score = self.calculate_engagement_score(post['engagement'])
        posted_at = datetime.fromisoformat(post['posted_at'])
        age_hours = (datetime.now() - posted_at).total_seconds() / 3600

        # Categorize performance
        if age_hours < 24:
            category = 'too_early'
        elif score > 2.0:
            category = 'high_performing'
        elif score > 1.0:
            category = 'average'
        else:
            category = 'underperforming'

        return {
            'post_id': post['post_id'],
            'engagement_score': score,
            'performance_category': category,
            'age_hours': age_hours
        }

    def detect_anomalies(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalous posts based on engagement patterns.

        Args:
            posts: List of post data dictionaries

        Returns:
            List of anomalous posts with analysis
        """
        if not posts:
            return []

        scores = [self.calculate_engagement_score(post['engagement']) for post in posts]
        mean_score = sum(scores) / len(scores)
        std_dev = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

        anomalies = []
        for post, score in zip(posts, scores):
            if abs(score - mean_score) > self.anomaly_threshold * std_dev:
                anomalies.append({
                    'post_id': post['post_id'],
                    'score': score,
                    'deviation': abs(score - mean_score) / std_dev
                })

        return anomalies

    def analyze_ab_test_results(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze A/B test results.

        Args:
            test_data: A/B test data dictionary

        Returns:
            Dictionary containing test analysis results
        """
        if not test_data:
            logger.warning("Insufficient data")
            return {'winning_variant': None, 'confidence': 0.0}

        variants = test_data.get('variants', {})
        if len(variants) < 2:
            return {'winning_variant': None, 'confidence': 0.0}

        # Calculate performance for each variant
        variant_scores = {}
        for variant, data in variants.items():
            if data.get('posts'):
                scores = [self.calculate_engagement_score(post['engagement'])
                         for post in data['posts']]
                variant_scores[variant] = sum(scores) / len(scores)

        if not variant_scores:
            return {'winning_variant': None, 'confidence': 0.0}

        # Find winning variant
        winner = max(variant_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, (winner[1] - min(variant_scores.values())) /
                        max(variant_scores.values()))

        return {
            'winning_variant': winner[0],
            'confidence': confidence
        }

    def generate_recommendations(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content and timing recommendations based on analysis.

        Args:
            engagement_data: Engagement data dictionary

        Returns:
            Dictionary containing recommendations
        """
        content_perf = self.analyze_content_type_performance(engagement_data)
        topic_perf = self.analyze_topic_performance(engagement_data)
        timing_perf = self.analyze_timing_performance(engagement_data)
        trends = self.analyze_engagement_trends(engagement_data)

        recommendations = {
            'content_recommendations': {
                'recommended_types': content_perf['best_types'],
                'recommended_topics': topic_perf['best_topics']
            },
            'timing_recommendations': {
                'best_days': timing_perf['best_days'],
                'best_hours': timing_perf['best_hours']
            },
            'trend_insights': {
                'direction': trends['trend_direction'],
                'strength': trends['trend_strength']
            }
        }

        return recommendations 