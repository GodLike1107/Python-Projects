import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Jira details from environment variables
JIRA_DOMAIN = os.getenv("JIRA_BASE_URL")
EMAIL = os.getenv("JIRA_EMAIL")
API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_API_URL = f"{JIRA_DOMAIN}/rest/api/3/issue/"

# Headers for authentication
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Basic {requests.auth._basic_auth_str(EMAIL, API_TOKEN)}"
}

# Function to fetch Jira issue details
def get_issue(issue_key):
    response = requests.get(f"{JIRA_API_URL}{issue_key}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch issue: {response.status_code}"}

# Function to update Jira issue status
def update_issue_status(issue_key, new_status):
    transition_data = {"transition": {"id": str(new_status)}}
    response = requests.post(f"{JIRA_API_URL}{issue_key}/transitions",
                             headers=HEADERS, json=transition_data)
    if response.status_code == 204:
        return {"success": "Issue status updated successfully"}
    else:
        return {"error": f"Failed to update status: {response.status_code}"}