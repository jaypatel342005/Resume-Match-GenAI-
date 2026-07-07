import warnings
import json
import os
import argparse
from dotenv import load_dotenv
from pipeline import pipeline

warnings.filterwarnings("ignore")
load_dotenv()

parser = argparse.ArgumentParser(description="Match a Resume to a Job Description using GenAI.")
parser.add_argument("--resume", required=True)
parser.add_argument("--jd", required=True)

args = parser.parse_args()

try:
    result = pipeline.invoke({"resume": args.resume, "jd": args.jd})
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Pipeline error: {e}")
