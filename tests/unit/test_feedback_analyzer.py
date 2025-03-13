"""Unit tests for feedback analyzer module."""
import pytest
from datetime import datetime, timedelta
from src.feedback.analyzer import FeedbackAnalyzer

@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return {
        'feedback': {
            'analysis': {
                'min_posts': 5,
                'anomaly_threshold': 2.0,
                'trend_window_days': 30
            },
            'metrics': {
                'engagement': {
                    'likes_weight': 1.0,
                    'comments_weight': 2.0,
                    'shares_weight': 3.0
                }
            }
        }
    }

@pytest.fixture
def mock_datetime(mocker):
    """Mock datetime for consistent testing."""
    mock_now = datetime(2025, 3, 13, 12, 0, 0)
    datetime_mock = mocker.patch('src.feedback.analyzer.datetime', wraps=datetime)
    datetime_mock.now.return_value = mock_now
    return mock_now

@pytest.fixture
def mock_engagement_data():
    """Mock engagement data fixture."""
    base_date = datetime(2025, 3, 13, 12, 0, 0)
    return {
        'posts': [
            {
                'post_id': 'post_1',
                'posted_at': (base_date - timedelta(days=1)).isoformat(),
                'engagement': {
                    'likes': 1000,
                    'comments': 500,
                    'shares': 200,
                    'impressions': 1000
                }
            },
            {
                'post_id': 'post_2',
                'posted_at': (base_date - timedelta(days=2)).isoformat(),
                'engagement': {
                    'likes': 800,
                    'comments': 400,
                    'shares': 150,
                    'impressions': 800
                }
            }
        ],
        'day_performance': {
            'monday': {'engagement_rate': 2.5, 'posts': 5},
            'friday': {'engagement_rate': 2.0, 'posts': 2}
        },
        'topic_performance': {
            'AI': {'engagement_rate': 2.5, 'posts': 4},
            'Deep Learning': {'engagement_rate': 3.0, 'posts': 6}
        }
    }

@pytest.mark.unit
def test_feedback_analyzer_initialization(test_config):
    """Test feedback analyzer initialization."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    assert analyzer.config == test_config['feedback']
    assert analyzer.min_posts == test_config['feedback']['analysis']['min_posts']
    assert analyzer.anomaly_threshold == test_config['feedback']['analysis']['anomaly_threshold']

@pytest.mark.unit
def test_calculate_engagement_score(test_config):
    """Test engagement score calculation."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    engagement = {
        'likes': 100,
        'comments': 50,
        'shares': 25,
        'impressions': 1000
    }
    score = analyzer.calculate_engagement_score(engagement)
    expected_score = (100 * 1.0 + 50 * 2.0 + 25 * 3.0) / 1000
    assert score == pytest.approx(expected_score)

@pytest.mark.unit
def test_analyze_content_type_performance(test_config, caplog):
    """Test content type performance analysis."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    result = analyzer.analyze_content_type_performance({})
    assert isinstance(result, dict)
    assert 'best_types' in result
    assert len(result['best_types']) == 0

@pytest.mark.unit
def test_analyze_topic_performance(test_config, caplog):
    """Test topic performance analysis."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    result = analyzer.analyze_topic_performance({})
    assert isinstance(result, dict)
    assert 'best_topics' in result
    assert len(result['best_topics']) == 0

@pytest.mark.unit
def test_analyze_engagement_trends(test_config, caplog, mock_engagement_data):
    """Test analyzing engagement trends."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    trends = analyzer.analyze_engagement_trends(mock_engagement_data)
    assert isinstance(trends, dict)
    assert 'trend_direction' in trends
    assert 'trend_strength' in trends

@pytest.mark.unit
def test_analyze_post_performance(test_config, mock_engagement_data, caplog, mock_datetime):
    """Test post performance analysis."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    performance = analyzer.analyze_post_performance(mock_engagement_data['posts'][0])
    assert isinstance(performance, dict)
    assert 'engagement_score' in performance
    assert 'performance_category' in performance

@pytest.mark.unit
def test_detect_anomalies(test_config, caplog):
    """Test anomaly detection."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    anomalies = analyzer.detect_anomalies([])
    assert isinstance(anomalies, list)
    assert len(anomalies) == 0

@pytest.mark.unit
def test_analyze_ab_test_results(test_config, caplog):
    """Test A/B test results analysis."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    results = analyzer.analyze_ab_test_results({})
    assert isinstance(results, dict)
    assert 'winning_variant' in results
    assert results['winning_variant'] is None

@pytest.mark.unit
def test_generate_recommendations(test_config, caplog, mock_engagement_data):
    """Test generating recommendations."""
    analyzer = FeedbackAnalyzer(test_config['feedback'])
    recommendations = analyzer.generate_recommendations(mock_engagement_data)
    assert isinstance(recommendations, dict)
    assert 'content_recommendations' in recommendations
    assert 'timing_recommendations' in recommendations 