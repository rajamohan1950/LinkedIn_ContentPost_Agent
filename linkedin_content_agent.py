
import os
import random
from dotenv import load_dotenv
from linkedin_api import Linkedin

load_dotenv()

# Authenticate LinkedIn
USERNAME = os.getenv("LINKEDIN_USERNAME")
PASSWORD = os.getenv("LINKEDIN_PASSWORD")

api = Linkedin(USERNAME, PASSWORD)

# Load topics
with open('topics.txt', 'r') as file:
    topics = [line.strip() for line in file if line.strip()]

# Content generation
def generate_content(topic):
    return (f"🚀 **{topic} Explained!**\n\n"
            f"At MAANG companies, {topic} empowers scalable, high-performance AI-driven products. "
            f"Let's connect and discuss how {topic} can transform your business!\n\n"
            f"#AI #ML #Cloud #EnterpriseAI #ThoughtLeadership")

# Select random topic for instant posting
selected_topic = random.choice(topics)
content = generate_content(selected_topic)

print(f"\n✅ Posting LinkedIn content instantly on topic: {selected_topic}\n")

# Post to LinkedIn
try:
    api.submit_share(content)
    print("🎉 Successfully posted to LinkedIn!")
except Exception as e:
    print(f"❌ Failed to post content: {e}")


