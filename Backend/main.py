import warnings
warnings.filterwarnings("ignore")

import json
import os
import argparse
from dotenv import load_dotenv

load_dotenv() 

from pipeline import pipeline 

parser = argparse.ArgumentParser(description="Match a Resume to a Job Description using GenAI.")
parser.add_argument("--resume", required=True, help="Path to resume PDF/text file or raw resume text")
parser.add_argument("--jd", required=True, help="Path to job description PDF/text file or raw job description text")

args = parser.parse_args()

print("Running pipeline...\n")

try:
    result = pipeline.invoke({"resume": args.resume, "jd": args.jd})
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error running pipeline: {e}")


