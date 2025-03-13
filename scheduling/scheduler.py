"""
Content Scheduling Module

This module is responsible for scheduling content for posting:
- Determines optimal posting times
- Manages posting frequency
- Handles approval workflows
- Learns from engagement data
"""

import logging
import random
import os
import json
from typing import Dict, List, Any, Optional
import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml

logger = logging.getLogger(__name__)

class ContentScheduler:
    """Schedules content for posting to LinkedIn."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the content scheduler with configuration.
        
        Args:
            config: Configuration dictionary for scheduling
        """
        self.config = config
        self.posting_frequency = config.get('posting_frequency', {})
        self.optimal_timing = config.get('optimal_timing', {})
        self.time_slots = config.get('time_slots', [])
        self.approval_workflow = config.get('approval_workflow', {})
        
        # Load historical engagement data if available
        self.engagement_data = self._load_engagement_data()
        
        # Load email credentials for approval notifications
        self.email_credentials = self._load_email_credentials()
        
        logger.info("Content scheduler initialized")
    
    def _load_engagement_data(self) -> Dict[str, Any]:
        """
        Load historical engagement data from file.
        
        Returns:
            Dictionary containing engagement data
        """
        try:
            data_path = os.path.join('data', 'engagement.json')
            
            if os.path.exists(data_path):
                with open(data_path, 'r') as file:
                    return json.load(file)
            else:
                logger.info(f"Engagement data file not found: {data_path}")
                return {'time_slots': {}, 'days': {}, 'posts': []}
                
        except Exception as e:
            logger.error(f"Error loading engagement data: {e}")
            return {'time_slots': {}, 'days': {}, 'posts': []}
    
    def _load_email_credentials(self) -> Dict[str, Any]:
        """
        Load email credentials for approval notifications.
        
        Returns:
            Dictionary containing email credentials
        """
        try:
            credentials_path = os.path.join('config', 'credentials.yml')
            
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as file:
                    all_credentials = yaml.safe_load(file)
                    return all_credentials.get('email', {})
            else:
                logger.warning(f"Credentials file not found: {credentials_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading email credentials: {e}")
            return {}
    
    def schedule(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Schedule content items for posting.
        
        Args:
            content_items: List of content items to schedule
            
        Returns:
            List of scheduled content items with posting times
        """
        logger.info(f"Scheduling {len(content_items)} content items")
        
        scheduled_items = []
        
        # Get the current time
        now = datetime.datetime.now()
        
        # Calculate the next available posting times
        next_posting_times = self._get_next_posting_times(len(content_items), now)
        
        # Schedule each content item
        for i, content in enumerate(content_items):
            # Get the next posting time
            if i < len(next_posting_times):
                posting_time = next_posting_times[i]
            else:
                # Fallback: schedule for tomorrow at a random time
                tomorrow = now + datetime.timedelta(days=1)
                posting_time = datetime.datetime(
                    tomorrow.year, tomorrow.month, tomorrow.day,
                    random.randint(9, 17), random.randint(0, 59)
                )
            
            # Add scheduling information to the content
            scheduled_content = content.copy()
            scheduled_content.update({
                'scheduled_time': posting_time.isoformat(),
                'status': 'scheduled',
                'approval_status': 'pending' if self.approval_workflow.get('enabled', True) else 'approved',
                'scheduled_at': now.isoformat(),
            })
            
            # Send approval notification if enabled
            if self.approval_workflow.get('enabled', True):
                self._send_approval_notification(scheduled_content)
            
            scheduled_items.append(scheduled_content)
            logger.info(f"Scheduled content '{content['title']}' for {posting_time.isoformat()}")
        
        return scheduled_items
    
    def _get_next_posting_times(self, num_items: int, start_time: datetime.datetime) -> List[datetime.datetime]:
        """
        Get the next available posting times.
        
        Args:
            num_items: Number of content items to schedule
            start_time: Starting time for scheduling
            
        Returns:
            List of posting times
        """
        posting_times = []
        
        # Get posting frequency settings
        posts_per_week = self.posting_frequency.get('posts_per_week', 3)
        min_hours_between_posts = self.posting_frequency.get('min_hours_between_posts', 24)
        
        # Convert to posts per day
        posts_per_day = posts_per_week / 7
        
        # Calculate the time delta between posts
        hours_between_posts = max(min_hours_between_posts, 24 / posts_per_day)
        
        # Get the next posting time
        next_time = start_time
        
        # If optimal timing is enabled, use learned optimal times
        if self.optimal_timing.get('enabled', True) and self.engagement_data.get('time_slots'):
            # Use the optimal time slots based on historical engagement
            optimal_slots = self._get_optimal_time_slots(num_items)
            
            for slot in optimal_slots:
                day_name = slot['day'].lower()
                hour = slot['hour']
                
                # Find the next occurrence of this day
                days_ahead = self._days_ahead_to(day_name, next_time.weekday())
                next_day = next_time + datetime.timedelta(days=days_ahead)
                
                # Create the posting time
                posting_time = datetime.datetime(
                    next_day.year, next_day.month, next_day.day,
                    hour, random.randint(0, 59)
                )
                
                # If the posting time is in the past, move to next week
                if posting_time <= next_time:
                    posting_time += datetime.timedelta(days=7)
                
                posting_times.append(posting_time)
                
                # Update next_time to ensure minimum spacing between posts
                next_time = posting_time + datetime.timedelta(hours=min_hours_between_posts)
        else:
            # Use configured time slots
            for i in range(num_items):
                # Find the next available time slot
                slot = self._get_next_time_slot(next_time)
                
                if slot:
                    day_name = slot['day'].lower()
                    hour = int(slot['slots'][0].split(':')[0])  # Use the first available hour
                    
                    # Find the next occurrence of this day
                    days_ahead = self._days_ahead_to(day_name, next_time.weekday())
                    next_day = next_time + datetime.timedelta(days=days_ahead)
                    
                    # Create the posting time
                    posting_time = datetime.datetime(
                        next_day.year, next_day.month, next_day.day,
                        hour, random.randint(0, 59)
                    )
                    
                    # If the posting time is in the past, move to next week
                    if posting_time <= next_time:
                        posting_time += datetime.timedelta(days=7)
                else:
                    # Fallback: use the next day at a business hour
                    next_day = next_time + datetime.timedelta(days=1)
                    posting_time = datetime.datetime(
                        next_day.year, next_day.month, next_day.day,
                        random.randint(9, 17), random.randint(0, 59)
                    )
                
                posting_times.append(posting_time)
                
                # Update next_time to ensure minimum spacing between posts
                next_time = posting_time + datetime.timedelta(hours=min_hours_between_posts)
        
        return posting_times
    
    def _days_ahead_to(self, target_day: str, current_day: int) -> int:
        """
        Calculate the number of days ahead to reach the target day.
        
        Args:
            target_day: Target day name (monday, tuesday, etc.)
            current_day: Current day as an integer (0=Monday, 6=Sunday)
            
        Returns:
            Number of days ahead
        """
        # Convert day name to integer (0=Monday, 6=Sunday)
        day_mapping = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_day_num = day_mapping.get(target_day.lower(), 0)
        
        # Calculate days ahead
        days_ahead = (target_day_num - current_day) % 7
        
        # If it's the same day but later, use today
        if days_ahead == 0:
            days_ahead = 0
        
        return days_ahead
    
    def _get_next_time_slot(self, start_time: datetime.datetime) -> Optional[Dict[str, Any]]:
        """
        Get the next available time slot from configuration.
        
        Args:
            start_time: Starting time for scheduling
            
        Returns:
            Dictionary containing day and slots, or None if no slots are available
        """
        if not self.time_slots:
            return None
        
        # Get the current day of the week (0=Monday, 6=Sunday)
        current_day = start_time.weekday()
        current_hour = start_time.hour
        
        # Check each day starting from the current day
        for i in range(7):  # Check all 7 days of the week
            day_offset = (current_day + i) % 7
            
            # Convert day number to name
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = day_names[day_offset]
            
            # Find the time slot for this day
            for slot in self.time_slots:
                if slot.get('day', '').lower() == day_name:
                    # If it's the current day, filter out past hours
                    if i == 0:  # Current day
                        future_slots = [s for s in slot.get('slots', []) if int(s.split(':')[0]) > current_hour]
                        if future_slots:
                            return {'day': day_name, 'slots': future_slots}
                    else:  # Future day
                        return slot
        
        # If no slots are found, return None
        return None
    
    def _get_optimal_time_slots(self, num_slots: int) -> List[Dict[str, Any]]:
        """
        Get the optimal time slots based on historical engagement.
        
        Args:
            num_slots: Number of slots to return
            
        Returns:
            List of optimal time slots
        """
        # Get engagement data for time slots
        time_slot_engagement = self.engagement_data.get('time_slots', {})
        
        # Convert to a list of (day, hour, engagement) tuples
        slot_data = []
        for day, hours in time_slot_engagement.items():
            for hour, engagement in hours.items():
                slot_data.append({
                    'day': day,
                    'hour': int(hour),
                    'engagement': engagement
                })
        
        # Sort by engagement (highest first)
        slot_data.sort(key=lambda x: x['engagement'], reverse=True)
        
        # If we have enough data, use it
        if len(slot_data) >= num_slots:
            return slot_data[:num_slots]
        
        # Otherwise, fill in with configured time slots
        result = slot_data.copy()
        
        # Add configured time slots that aren't already in the result
        for slot in self.time_slots:
            day = slot.get('day', '').lower()
            for time_str in slot.get('slots', []):
                hour = int(time_str.split(':')[0])
                
                # Check if this slot is already in the result
                if not any(s['day'] == day and s['hour'] == hour for s in result):
                    result.append({
                        'day': day,
                        'hour': hour,
                        'engagement': 0  # No historical data
                    })
                    
                    # If we have enough slots, stop
                    if len(result) >= num_slots:
                        break
            
            # If we have enough slots, stop
            if len(result) >= num_slots:
                break
        
        # If we still don't have enough slots, add some default business hours
        while len(result) < num_slots:
            # Use a weekday and business hour
            day = random.choice(['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])
            hour = random.randint(9, 17)
            
            # Check if this slot is already in the result
            if not any(s['day'] == day and s['hour'] == hour for s in result):
                result.append({
                    'day': day,
                    'hour': hour,
                    'engagement': 0  # No historical data
                })
        
        return result[:num_slots]
    
    def _send_approval_notification(self, content: Dict[str, Any]) -> bool:
        """
        Send an approval notification email.
        
        Args:
            content: Content to be approved
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.approval_workflow.get('enabled', True):
            return True
        
        # Get email settings
        smtp_server = self.email_credentials.get('smtp_server', '')
        smtp_port = self.email_credentials.get('smtp_port', 587)
        username = self.email_credentials.get('username', '')
        password = self.email_credentials.get('password', '')
        from_address = self.email_credentials.get('from_address', '')
        to_address = self.approval_workflow.get('notification_email', '')
        
        # If any required settings are missing, log a warning and return
        if not all([smtp_server, username, password, from_address, to_address]):
            logger.warning("Email credentials incomplete, skipping approval notification")
            return False
        
        try:
            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = from_address
            msg['To'] = to_address
            msg['Subject'] = f"LinkedIn Post Approval: {content['title']}"
            
            # Create the email body
            body = f"""
            <html>
            <body>
                <h2>LinkedIn Post Approval Request</h2>
                <p>A new LinkedIn post has been scheduled and requires your approval:</p>
                
                <h3>{content['title']}</h3>
                <p><strong>Scheduled Time:</strong> {content['scheduled_time']}</p>
                <p><strong>Content Type:</strong> {content['type']}</p>
                
                <h4>Content:</h4>
                <p>{content['content']}</p>
                
                <h4>Hashtags:</h4>
                <p>{' '.join(['#' + tag for tag in content['hashtags']])}</p>
                
                <p>Please approve or reject this post by replying to this email.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to the SMTP server and send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Approval notification sent for content: {content['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending approval notification: {e}")
            return False
    
    def update_engagement_data(self, post_results: Dict[str, Any]) -> None:
        """
        Update engagement data based on post results.
        
        Args:
            post_results: Results from a posted content item
        """
        if not post_results.get('success', False):
            return
        
        try:
            # Extract posting time and engagement metrics
            posted_at = post_results.get('posted_at', '')
            engagement = post_results.get('engagement', {})
            
            if not posted_at:
                return
            
            # Parse the posting time
            posted_time = datetime.datetime.fromisoformat(posted_at)
            
            # Calculate engagement score
            engagement_score = (
                engagement.get('likes', 0) * 1.0 +
                engagement.get('comments', 0) * 2.0 +
                engagement.get('shares', 0) * 3.0 +
                engagement.get('impressions', 0) * 0.1
            )
            
            # Get the day and hour
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day = day_names[posted_time.weekday()]
            hour = str(posted_time.hour)
            
            # Update time slot engagement data
            if 'time_slots' not in self.engagement_data:
                self.engagement_data['time_slots'] = {}
            
            if day not in self.engagement_data['time_slots']:
                self.engagement_data['time_slots'][day] = {}
            
            if hour not in self.engagement_data['time_slots'][day]:
                self.engagement_data['time_slots'][day][hour] = 0
            
            # Update with exponential moving average
            alpha = self.optimal_timing.get('learning_rate', 0.1)
            current_value = self.engagement_data['time_slots'][day][hour]
            new_value = (1 - alpha) * current_value + alpha * engagement_score
            self.engagement_data['time_slots'][day][hour] = new_value
            
            # Update day engagement data
            if 'days' not in self.engagement_data:
                self.engagement_data['days'] = {}
            
            if day not in self.engagement_data['days']:
                self.engagement_data['days'][day] = 0
            
            current_value = self.engagement_data['days'][day]
            new_value = (1 - alpha) * current_value + alpha * engagement_score
            self.engagement_data['days'][day] = new_value
            
            # Add post to history
            if 'posts' not in self.engagement_data:
                self.engagement_data['posts'] = []
            
            post_data = {
                'post_id': post_results.get('post_id', ''),
                'posted_at': posted_at,
                'day': day,
                'hour': hour,
                'engagement_score': engagement_score,
                'engagement': engagement,
            }
            
            self.engagement_data['posts'].append(post_data)
            
            # Limit the number of stored posts
            max_posts = 100
            if len(self.engagement_data['posts']) > max_posts:
                self.engagement_data['posts'] = self.engagement_data['posts'][-max_posts:]
            
            # Save the updated engagement data
            self._save_engagement_data()
            
            logger.info(f"Updated engagement data for post at {day} {hour}:00")
            
        except Exception as e:
            logger.error(f"Error updating engagement data: {e}")
    
    def _save_engagement_data(self) -> None:
        """Save engagement data to file."""
        try:
            # Ensure the data directory exists
            os.makedirs('data', exist_ok=True)
            
            data_path = os.path.join('data', 'engagement.json')
            
            with open(data_path, 'w') as file:
                json.dump(self.engagement_data, file, indent=2)
            
            logger.info(f"Saved engagement data to {data_path}")
            
        except Exception as e:
            logger.error(f"Error saving engagement data: {e}")
    
    def get_optimal_posting_times(self, num_times: int = 5) -> List[Dict[str, Any]]:
        """
        Get the optimal posting times based on historical engagement.
        
        Args:
            num_times: Number of optimal times to return
            
        Returns:
            List of optimal posting times with metadata
        """
        return self._get_optimal_time_slots(num_times) 