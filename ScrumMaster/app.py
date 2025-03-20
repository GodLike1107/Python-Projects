import gradio as gr
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Jira credentials (Use environment variables for security)
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# Jira API URL
JIRA_API_URL = f"{JIRA_BASE_URL}/rest/api/3/issue/"

# Headers for authentication
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
AUTH = (JIRA_EMAIL, JIRA_API_TOKEN)

# Store search history
search_history = []


def get_jira_issue(issue_id):
    response = requests.get(f"{JIRA_API_URL}{issue_id}", headers=HEADERS, auth=AUTH)

    if response.status_code == 200:
        issue_data = response.json()
        issue_summary = issue_data["fields"].get("summary", "No Summary")
        issue_type = issue_data["fields"].get("issuetype", {}).get("name", "Unknown")
        issue_status = issue_data["fields"].get("status", {}).get("name", "Unknown")
        issue_priority = issue_data["fields"].get("priority", {}).get("name", "Not Set")
        issue_assignee = issue_data["fields"].get("assignee")
        issue_assignee_name = issue_assignee.get("displayName", "Unassigned") if issue_assignee else "Unassigned"
        issue_description = issue_data["fields"].get("description", {}).get("content", [])
        issue_description_text = " ".join([p["text"] for p in issue_description[0]["content"]]) if isinstance(
            issue_description, list) and issue_description else "No description available"

        # Store in history and update UI
        search_history.append(issue_id)
        history_box.value = "\n".join(search_history)

        return f"""
        🔖 **Summary:** `{issue_summary}`  
        🔵 **Issue Type:** `{issue_type}`  
        ✅ **Status:** `{issue_status}`  
        🔥 **Priority:** `{issue_priority}`  
        👤 **Assignee:** `{issue_assignee_name}`  
        📝 **Description:** `{issue_description_text}`
        """

    else:
        return f"❌ Error: {response.status_code}, {response.text}"


def create_jira_issue(summary, description, issue_type="Bug"):
    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": {"type": "doc", "version": 1,
                            "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]},
            "issuetype": {"name": issue_type}
        }
    }

    response = requests.post(JIRA_API_URL, headers=HEADERS, auth=AUTH, json=payload)
    if response.status_code == 201:
        return "✅ Issue Created Successfully!"
    else:
        return f"❌ Error: {response.status_code}, {response.text}"


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# � Jira Issue Manager")

    with gr.Tab("🔍 Fetch Issue Details"):
        gr.Markdown("## 🔍 Fetch Jira Issue Details")
        with gr.Row():
            issue_input = gr.Textbox(label="Enter Jira Issue ID (e.g., SCRUM-2)", placeholder="SCRUM-2")
            fetch_button = gr.Button("🔍 Fetch Issue Details", variant="primary")
        output = gr.Markdown()
        fetch_button.click(get_jira_issue, inputs=issue_input, outputs=output)

        gr.Markdown("### 📜 Search History")
        history_box = gr.Textbox(value="\n".join(search_history), interactive=False, lines=5, label="Search History")

    with gr.Tab("🆕 Create New Issue"):
        gr.Markdown("## 🆕 Create New Jira Issue")
        issue_summary = gr.Textbox(label="Summary", placeholder="Enter a brief summary of the issue")
        issue_description = gr.Textbox(label="Description", lines=3, placeholder="Describe the issue in detail")
        issue_type = gr.Dropdown(["Bug", "Task", "Story"], label="Issue Type", value="Bug")
        create_button = gr.Button("➕ Create Issue", variant="primary")
        create_output = gr.Markdown()
        create_button.click(create_jira_issue, inputs=[issue_summary, issue_description, issue_type], outputs=create_output)

    gr.Markdown("### ℹ️ About")
    gr.Markdown("This application allows you to fetch and create Jira issues. Use the tabs above to navigate between features.")

# Run the app
if __name__ == "__main__":
    app.launch()