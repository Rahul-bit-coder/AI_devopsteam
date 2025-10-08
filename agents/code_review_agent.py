from pydantic import BaseModel
from pydantic_ai import Agent  # Replace with actual import if different
from utils.groq_client import GROQClient
from models.groq_models import CodeReviewRequest, CodeReviewFeedback
import os
import requests
from typing import Optional, Any

# Lazy/optional import for GitHub client to avoid hard dependency during local reviews
try:
    from github import Github as _Github
except Exception:  # Module not present or shadowed by wrong package
    _Github = None

class CodeReviewConfig(BaseModel):
    """
    Configuration settings for the Code Review agent.
    
    Attributes:
        model (str): The LLM model to use for code review (default: llama3-8b-8192)
        groq_api_endpoint (str): GROQ API endpoint URL
        groq_api_key (str): Authentication key for GROQ API
        github_token (str): GitHub authentication token
        repo_name (str): GitHub repository name in format "username/repo"
        pull_request_number (int): PR number to review
    """
    model: str = "llama3-8b-8192"  # Default model for code review
    groq_api_endpoint: str
    groq_api_key: str
    github_token: str
    repo_name: str
    pull_request_number: int

class CodeReviewAgent(Agent):
    """
    An AI agent that performs automated code reviews on GitHub pull requests.
    
    This agent analyzes Python files in pull requests, provides feedback on code quality,
    and posts detailed review comments directly to GitHub.
    """

    def __init__(self, config: CodeReviewConfig):
        """
        Initialize the Code Review agent with necessary clients and configuration.
        
        Args:
            config (CodeReviewConfig): Configuration object containing API keys and settings
        """
        super().__init__()  # Don't pass config to parent
        self.config = config
        self.groq_client = GROQClient(
            api_endpoint=config.groq_api_endpoint,
            api_key=config.groq_api_key
        )
        # Initialize GitHub client only if library is available and token provided
        self.github_client = None
        if _Github is not None and self.config.github_token:
            try:
                self.github_client = _Github(self.config.github_token)
            except Exception:
                self.github_client = None

    def fetch_pull_request_files(self):
        """
        Retrieve the files modified in the specified pull request.
        
        Returns:
            PaginatedList: List of files modified in the pull request
        """
        if not self.github_client:
            raise RuntimeError("GitHub client unavailable. Install PyGithub and set GH_TOKEN, repo_name, pull_request_number.")
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)
        files = pull_request.get_files()
        return files

    def perform_code_review(self):
        """
        Analyze modified Python files in the pull request and generate review feedback.
        
        The method:
        1. Fetches modified files from the pull request
        2. Analyzes Python files using the GROQ API
        3. Generates detailed feedback for each file
        
        Returns:
            list: List of dictionaries containing feedback for each reviewed file
                 Including issues found, suggestions, and overall quality scores
        """
        files = self.fetch_pull_request_files()
        feedback = []

        include_exts = ('.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css')
        for file in files:
            if file.filename.endswith(include_exts):
                # Fetch actual file content via raw_url
                file_content_text = ""
                try:
                    if hasattr(file, 'raw_url') and file.raw_url:
                        r = requests.get(file.raw_url, timeout=15)
                        r.raise_for_status()
                        file_content_text = r.text
                except Exception:
                    file_content_text = ""

                code_review_request = CodeReviewRequest(
                    file_name=file.filename,
                    file_content=file_content_text,
                    diff=getattr(file, 'patch', '') or ''
                )
                try:
                    review_feedback = self.groq_client.send_code_review_request(
                        model_id=self.config.model,
                        code_review_request=code_review_request
                    )
                    feedback.append({
                        "file": file.filename,
                        "issues": review_feedback.issues,
                        "suggestions": review_feedback.suggestions,
                        "overall_quality": review_feedback.overall_quality
                    })
                except Exception as e:
                    feedback.append({
                        "file": file.filename,
                        "error": str(e)
                    })

        return feedback

    def post_feedback_to_github(self, feedback):
        """
        Post the code review feedback as comments on the GitHub pull request.
        
        Args:
            feedback (list): List of feedback dictionaries for each reviewed file
                           containing issues, suggestions, and quality scores
        """
        if not self.github_client:
            raise RuntimeError("GitHub client unavailable. Install PyGithub and set GH_TOKEN, repo_name, pull_request_number.")
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)

        for file_feedback in feedback:
            if "error" in file_feedback:
                # Handle error cases with warning message
                comment = f"⚠️ **Code Review Error**: {file_feedback['error']}"
            else:
                # Format successful review feedback
                issues = "\n".join([f"- {issue['description']}" for issue in file_feedback['issues']])
                suggestions = "\n".join([f"- {suggestion}" for suggestion in file_feedback['suggestions']])
                overall = file_feedback['overall_quality']

                comment = (
                    f"### 📝 Code Review for `{file_feedback['file']}`\n\n"
                    f"**Overall Quality**: {overall}\n\n"
                    f"**Issues Found**:\n{issues}\n\n"
                    f"**Suggestions**:\n{suggestions}"
                )
            # Post the comment on the pull request
            pull_request.create_issue_comment(comment)

    def run(self):
        """
        Execute the main workflow of the code review agent.
        
        This method:
        1. Performs code review on the pull request files
        2. Posts the feedback to GitHub
        3. Returns the complete feedback data
        
        Returns:
            list: Complete feedback data for all reviewed files
        """
        feedback = self.perform_code_review()
        self.post_feedback_to_github(feedback)
        return feedback

    def review_local_file(self, file_path: str):
        """
        Review a local file (any text file, e.g., HTML) without using GitHub.
        Returns a structured feedback dict from the GROQ API.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"file": file_path, "error": f"Failed to read file: {str(e)}"}

        request = CodeReviewRequest(
            file_name=os.path.basename(file_path),
            file_content=content,
            diff=""  # No diff context for local review
        )
        try:
            feedback: CodeReviewFeedback = self.groq_client.send_code_review_request(
                model_id=self.config.model,
                code_review_request=request
            )
            return {
                "file": file_path,
                "issues": feedback.issues,
                "suggestions": feedback.suggestions,
                "overall_quality": feedback.overall_quality
            }
        except Exception as e:
            return {"file": file_path, "error": str(e)}