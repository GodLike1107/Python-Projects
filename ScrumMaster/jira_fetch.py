import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

# Jira API credentials
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")


def fetch_active_sprint_issues():
    """Fetches all issues from the active sprint in Jira."""

    # Jira JQL Query to get active sprint issues
    jql_query = f'project="{JIRA_PROJECT_KEY}" AND sprint in openSprints()'

    url = f"{JIRA_BASE_URL}/rest/api/3/search"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}

    params = {
        "jql": jql_query,
        "fields": ["summary", "status", "assignee"],
        "maxResults": 50  # Adjust as needed
    }

    response = requests.get(url, headers=headers, auth=auth, params=params)

    if response.status_code == 200:
        issues = response.json().get("issues", [])
        print(f"✅ Fetched {len(issues)} issues from the active sprint!")
        for issue in issues:
            key = issue["key"]
            summary = issue["fields"]["summary"]
            status = issue["fields"]["status"]["name"]
            assignee = issue["fields"]["assignee"]["displayName"] if issue["fields"]["assignee"] else "Unassigned"
            print(f"🔹 {key}: {summary} [{status}] - Assigned to: {assignee}")
        return issues
    else:
        print("❌ Failed to fetch sprint issues:", response.text)
        return None


# Run the function
if __name__ == "__main__":
    fetch_active_sprint_issues()
