"""Script to post content to LinkedIn."""
import os
import logging
from dotenv import load_dotenv
from linkedin_api import Linkedin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def post_to_linkedin():
    """Post content to LinkedIn."""
    # Load environment variables
    load_dotenv()
    
    # Get LinkedIn credentials
    username = os.getenv('LINKEDIN_USERNAME')
    password = os.getenv('LINKEDIN_PASSWORD')
    
    if not username or not password:
        logger.error("LinkedIn credentials not found in environment variables")
        return
    
    try:
        # Initialize LinkedIn API client
        api = Linkedin(username, password)
        
        # Example post content
        post_content = {
            'content': {
                'contentEntities': [{
                    'location': 'urn:li:geo:103644278',
                    'title': 'Understanding Deep Learning Fundamentals',
                    'contentType': 'ARTICLE',
                    'content': 'Deep learning is revolutionizing the way we approach artificial intelligence. Here are some key fundamentals:\n\n1. Neural Networks: The building blocks of deep learning\n2. Backpropagation: How neural networks learn\n3. Activation Functions: Bringing non-linearity to networks\n\nWhat aspects of deep learning interest you the most? Share your thoughts below!',
                    'topics': ['AI', 'Deep Learning']
                }],
                'title': 'Understanding Deep Learning Fundamentals'
            },
            'distribution': {
                'linkedInDistributionTarget': {
                    'visibleToGuest': True
                }
            },
            'owner': 'urn:li:person:YOUR_PROFILE_ID',  # Replace with your profile ID
            'subject': 'Deep Learning Fundamentals',
            'text': {
                'text': 'Deep learning is revolutionizing the way we approach artificial intelligence. Here are some key fundamentals:\n\n1. Neural Networks: The building blocks of deep learning\n2. Backpropagation: How neural networks learn\n3. Activation Functions: Bringing non-linearity to networks\n\nWhat aspects of deep learning interest you the most? Share your thoughts below!'
            }
        }
        
        # Post to LinkedIn
        response = api.post(post_content)
        logger.info("Successfully posted to LinkedIn!")
        logger.info(f"Post ID: {response.get('id')}")
        
    except Exception as e:
        logger.error(f"Error posting to LinkedIn: {str(e)}")
        raise

if __name__ == "__main__":
    post_to_linkedin() 