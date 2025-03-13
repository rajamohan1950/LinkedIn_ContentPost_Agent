"""
LinkedIn API Integration Module

This module is responsible for interacting with the LinkedIn API:
- Authentication and authorization
- Posting content to LinkedIn
- Tracking engagement metrics
- Collecting analytics
"""

import logging
import os
import json
import time
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

class LinkedInClient:
    """Client for interacting with the LinkedIn API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LinkedIn client with configuration.
        
        Args:
            config: Configuration dictionary for LinkedIn API
        """
        self.config = config
        self.api_version = config.get('api_version', 'v2')
        self.oauth_config = config.get('oauth', {})
        self.post_settings = config.get('post_settings', {})
        
        # Load credentials
        self.credentials = self._load_credentials()
        
        # Set up API endpoints
        self.base_url = f"https://api.linkedin.com/rest"
        
        logger.info("LinkedIn client initialized")
    
    def _load_credentials(self) -> Dict[str, Any]:
        """
        Load LinkedIn API credentials from file.
        
        Returns:
            Dictionary containing credentials
        """
        try:
            credentials_path = os.path.join('config', 'credentials.yml')
            
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as file:
                    all_credentials = yaml.safe_load(file)
                    return all_credentials.get('linkedin', {})
            else:
                logger.warning(f"Credentials file not found: {credentials_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for LinkedIn API requests.
        
        Returns:
            Dictionary containing request headers
        """
        access_token = self.credentials.get('access_token', '')
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-Restli-Protocol-Version': '2.0.0',
            'LinkedIn-Version': self.api_version,
            'Content-Type': 'application/json',
        }
        
        return headers
    
    def _refresh_token_if_needed(self) -> bool:
        """
        Refresh the access token if it's expired or about to expire.
        
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        # Check if we have a refresh token
        refresh_token = self.credentials.get('refresh_token')
        if not refresh_token:
            logger.warning("No refresh token available")
            return False
        
        # In a real implementation, you would check token expiration
        # and refresh if needed using the OAuth 2.0 refresh token flow
        
        # For demonstration, assume token is still valid
        return True
    
    def post(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post content to LinkedIn.
        
        Args:
            content: Content to post
            
        Returns:
            Dictionary containing post results
        """
        logger.info(f"Posting content to LinkedIn: {content['title']}")
        
        # Ensure token is valid
        if not self._refresh_token_if_needed():
            logger.error("Failed to refresh token, cannot post content")
            return {'success': False, 'error': 'Authentication failed'}
        
        # Prepare post data
        post_data = self._prepare_post_data(content)
        
        # In a real implementation, you would make the API call
        # response = self._make_api_call('POST', '/posts', post_data)
        
        # For demonstration, simulate a successful post
        post_id = f"urn:li:share:{int(time.time())}"
        
        # Record post details for tracking
        post_result = {
            'success': True,
            'post_id': post_id,
            'content': content,
            'posted_at': datetime.now().isoformat(),
            'engagement': {
                'impressions': 0,
                'likes': 0,
                'comments': 0,
                'shares': 0,
            }
        }
        
        logger.info(f"Content posted successfully with ID: {post_id}")
        return post_result
    
    def _prepare_post_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for LinkedIn post.
        
        Args:
            content: Content to post
            
        Returns:
            Dictionary containing formatted post data
        """
        # Get author URN (person or organization)
        author = self.credentials.get('person_urn', self.credentials.get('organization_urn', ''))
        
        # Get visibility setting
        visibility = self.post_settings.get('visibility', 'connections-only')
        linkedin_visibility = 'PUBLIC' if visibility == 'public' else 'CONNECTIONS'
        
        # Prepare post content
        post_text = content.get('content', '')
        
        # Add hashtags if enabled
        if self.post_settings.get('include_hashtags', True):
            hashtags = content.get('hashtags', [])
            max_hashtags = self.post_settings.get('max_hashtags', 5)
            
            # Limit number of hashtags
            hashtags = hashtags[:max_hashtags]
            
            # Add hashtags to post if not already included
            if hashtags and not any(f"#{tag}" in post_text for tag in hashtags):
                hashtag_text = ' '.join([f"#{tag}" for tag in hashtags])
                post_text = f"{post_text}\n\n{hashtag_text}"
        
        # Prepare post data according to LinkedIn API format
        post_data = {
            "author": author,
            "commentary": post_text,
            "visibility": linkedin_visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": []
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False
        }
        
        # Add media if available
        if content.get('images'):
            # In a real implementation, you would upload images first
            # and then include the image URNs in the post
            pass
        
        return post_data
    
    def _make_api_call(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an API call to LinkedIn.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data (for POST, PUT, etc.)
            
        Returns:
            API response
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return {'success': False, 'error': f"Unsupported HTTP method: {method}"}
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse response
            if response.content:
                return response.json()
            else:
                return {'success': True}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_engagement_metrics(self, post_id: str) -> Dict[str, Any]:
        """
        Get engagement metrics for a post.
        
        Args:
            post_id: ID of the post
            
        Returns:
            Dictionary containing engagement metrics
        """
        logger.info(f"Getting engagement metrics for post: {post_id}")
        
        # Ensure token is valid
        if not self._refresh_token_if_needed():
            logger.error("Failed to refresh token, cannot get metrics")
            return {'success': False, 'error': 'Authentication failed'}
        
        # In a real implementation, you would make the API call
        # response = self._make_api_call('GET', f'/socialActions/{post_id}')
        
        # For demonstration, simulate engagement metrics
        # In a real implementation, these would come from the API
        metrics = {
            'success': True,
            'post_id': post_id,
            'engagement': {
                'impressions': 150 + int(time.time() % 100),
                'likes': 5 + int(time.time() % 10),
                'comments': 2 + int(time.time() % 5),
                'shares': 1 + int(time.time() % 3),
            },
            'retrieved_at': datetime.now().isoformat(),
        }
        
        logger.info(f"Retrieved engagement metrics for post: {post_id}")
        return metrics
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """
        Get profile statistics.
        
        Returns:
            Dictionary containing profile statistics
        """
        logger.info("Getting profile statistics")
        
        # Ensure token is valid
        if not self._refresh_token_if_needed():
            logger.error("Failed to refresh token, cannot get profile stats")
            return {'success': False, 'error': 'Authentication failed'}
        
        # In a real implementation, you would make the API call
        # response = self._make_api_call('GET', '/me')
        
        # For demonstration, simulate profile statistics
        # In a real implementation, these would come from the API
        stats = {
            'success': True,
            'profile_views': 50 + int(time.time() % 30),
            'follower_count': 500 + int(time.time() % 50),
            'connection_count': 1000 + int(time.time() % 100),
            'retrieved_at': datetime.now().isoformat(),
        }
        
        logger.info("Retrieved profile statistics")
        return stats 