
import warnings
warnings.filterwarnings("ignore")

import json
from dotenv import load_dotenv

load_dotenv() 

from pipeline import pipeline 
resume = """
John Doe — Python Developer

Skills: Python, FastAPI, Django, PostgreSQL, Redis, Docker, AWS S3, AWS Lambda

Experience:
- Backend Engineer at TechCorp (2021–Present)
  Built REST APIs with FastAPI deployed on AWS using Docker.

- Junior Developer at StartupXYZ (2019–2021)
  Developed Django REST APIs, worked with PostgreSQL and Redis.

Education: B.Sc Computer Science, State University, 2019
"""

jd = """
We are hiring a Senior Python Engineer.

Requirements:
- 4+ years Python experience
- FastAPI or Django
- PostgreSQL and Redis
- Docker and Kubernetes
- AWS (S3, Lambda, EC2)
- Machine learning (scikit-learn, pandas) is a plus
"""


print("Running pipeline...\n")

result = pipeline.invoke({"resume": resume, "jd": jd})

print(json.dumps(result, indent=2))
