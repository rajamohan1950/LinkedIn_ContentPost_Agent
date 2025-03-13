#!/usr/bin/env python3
"""
LinkedIn Deep Learning Thought Leadership Agent

This is the main entry point for the LinkedIn content posting agent.
It orchestrates the entire workflow from data collection to content posting.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data_collection.collector import DataCollector
from src.content_analysis.analyzer import ContentAnalyzer
from src.content_generation.generator import ContentGenerator
from src.linkedin_api.linkedin_client import LinkedInClient
from src.scheduling.scheduler import ContentScheduler
from src.feedback.analyzer import FeedbackAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LinkedIn Deep Learning Thought Leadership Agent')
    parser.add_argument('--config', type=str, default='config/config.yml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'generate', 'schedule', 'post', 'feedback', 'full'],
                        default='full', help='Operation mode')
    return parser.parse_args()

class LinkedInAgent:
    """Main agent class that orchestrates the workflow."""
    
    def __init__(self, config):
        """Initialize the agent with configuration."""
        self.config = config
        self.data_collector = DataCollector(config['data_collection'])
        self.content_analyzer = ContentAnalyzer(config['content_analysis'])
        self.content_generator = ContentGenerator(config['content_generation'])
        self.linkedin_client = LinkedInClient(config['linkedin_api'])
        self.scheduler = ContentScheduler(config['scheduling'])
        self.feedback_analyzer = FeedbackAnalyzer(config['feedback'])
        
        logger.info("LinkedIn Agent initialized")
    
    def collect_data(self):
        """Collect data from various sources."""
        logger.info("Starting data collection")
        return self.data_collector.collect()
    
    def analyze_content(self, data):
        """Analyze collected data for insights."""
        logger.info("Starting content analysis")
        return self.content_analyzer.analyze(data)
    
    def generate_content(self, insights):
        """Generate content based on insights."""
        logger.info("Starting content generation")
        return self.content_generator.generate(insights)
    
    def schedule_content(self, content):
        """Schedule content for posting."""
        logger.info("Scheduling content")
        return self.scheduler.schedule(content)
    
    def post_content(self, scheduled_content):
        """Post content to LinkedIn."""
        logger.info("Posting content to LinkedIn")
        return self.linkedin_client.post(scheduled_content)
    
    def analyze_feedback(self, post_results):
        """Analyze feedback and engagement."""
        logger.info("Analyzing feedback")
        return self.feedback_analyzer.analyze(post_results)
    
    def run_full_workflow(self):
        """Run the complete workflow from data collection to feedback analysis."""
        logger.info("Starting full workflow")
        
        # Collect data
        data = self.collect_data()
        
        # Analyze content
        insights = self.analyze_content(data)
        
        # Generate content
        content = self.generate_content(insights)
        
        # Schedule content
        scheduled_content = self.schedule_content(content)
        
        # Post content
        post_results = self.post_content(scheduled_content)
        
        # Analyze feedback
        feedback_results = self.analyze_feedback(post_results)
        
        logger.info("Full workflow completed")
        return feedback_results

def main():
    """Main function to run the agent."""
    args = parse_arguments()
    config = load_config(args.config)
    
    agent = LinkedInAgent(config)
    
    if args.mode == 'collect':
        agent.collect_data()
    elif args.mode == 'analyze':
        data = agent.collect_data()
        agent.analyze_content(data)
    elif args.mode == 'generate':
        data = agent.collect_data()
        insights = agent.analyze_content(data)
        agent.generate_content(insights)
    elif args.mode == 'schedule':
        data = agent.collect_data()
        insights = agent.analyze_content(data)
        content = agent.generate_content(insights)
        agent.schedule_content(content)
    elif args.mode == 'post':
        data = agent.collect_data()
        insights = agent.analyze_content(data)
        content = agent.generate_content(insights)
        scheduled_content = agent.schedule_content(content)
        agent.post_content(scheduled_content)
    elif args.mode == 'feedback':
        data = agent.collect_data()
        insights = agent.analyze_content(data)
        content = agent.generate_content(insights)
        scheduled_content = agent.schedule_content(content)
        post_results = agent.post_content(scheduled_content)
        agent.analyze_feedback(post_results)
    elif args.mode == 'full':
        agent.run_full_workflow()
    
    logger.info(f"Agent completed {args.mode} mode")

if __name__ == "__main__":
    main() 