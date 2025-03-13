# LinkedIn Content Post Agent

A Python-based tool for automating LinkedIn content posting and analyzing post performance.

## Features

- Automated LinkedIn content posting
- Post performance analysis
- Engagement metrics tracking
- Content type and topic performance analysis
- Timing optimization recommendations
- A/B testing capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LinkedIn_ContentPost_Agent.git
cd LinkedIn_ContentPost_Agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your LinkedIn credentials:
```
LINKEDIN_USERNAME=your_email@example.com
LINKEDIN_PASSWORD=your_password
```

## Usage

1. Post content to LinkedIn:
```bash
python post_to_linkedin.py
```

2. Analyze post performance:
```bash
python run.py
```

## Project Structure

```
LinkedIn_ContentPost_Agent/
├── src/
│   ├── feedback/
│   │   └── analyzer.py
│   └── ...
├── tests/
│   ├── unit/
│   │   └── test_feedback_analyzer.py
│   └── ...
├── post_to_linkedin.py
├── run.py
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security

- Never commit your `.env` file or expose your LinkedIn credentials
- Use environment variables for sensitive information
- Keep your dependencies up to date

## Disclaimer

This tool is for educational purposes only. Please comply with LinkedIn's terms of service and API usage guidelines. 