from pydantic import BaseModel
from pydantic_ai import Agent
from models.groq_models import DockerConfig
from utils.groq_client import GROQClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DockerfileConfig(BaseModel):
    """
    Configuration settings for the Dockerfile generator agent.
    
    Attributes:
        base_image (str): Base Docker image to use (e.g., nginx:alpine)
        expose_port (int): Port number to expose in the container
        copy_source (str): Source directory to copy into the container
        work_dir (str): Working directory inside the container
        groq_api_endpoint (str): GROQ API endpoint URL
        groq_api_key (str): Authentication key for GROQ API
    """
    base_image: str
    expose_port: int
    copy_source: str
    work_dir: str
    groq_api_endpoint: str
    groq_api_key: str


class DockerfileAgent(Agent):
    """
    An AI agent that generates and manages Dockerfile configurations.
    
    This agent can fetch configuration from GROQ's API and generate
    appropriate Dockerfile content based on the configuration.
    """

    def __init__(self, config: DockerfileConfig):
        """
        Initialize the Dockerfile agent with necessary configuration.
        
        Args:
            config (DockerfileConfig): Configuration object containing Docker and API settings
        """
        super().__init__()  # Call parent without config
        self.config = config
        self.groq_client = GROQClient(
            api_endpoint=config.groq_api_endpoint,
            api_key=config.groq_api_key
        )

    def fetch_config(self):
        """
        Fetch Dockerfile configuration from GROQ API.
        
        Queries the GROQ API for Docker configuration settings and updates
        the agent's configuration accordingly. Falls back to default values
        if the API request fails.
        """
        # Query GROQ API for Docker configuration
        groq_query = "*[_type == 'dockerConfig'][0]{baseImage, exposePort, copySource, workDir}"
        try:
            result = self.groq_client.query(groq_query)
        except Exception:
            result = None

        # Always enforce safe defaults; only override provided keys
        base = {
            "baseImage": self.config.base_image or "nginx:alpine",
            "exposePort": self.config.expose_port or 80,
            "copySource": self.config.copy_source or "./html",
            "workDir": self.config.work_dir or "/usr/share/nginx/html",
            "groqApiEndpoint": self.config.groq_api_endpoint,
            "groqApiKey": self.config.groq_api_key,
        }
        if isinstance(result, dict):
            base.update({k: v for k, v in result.items() if v not in (None, "")})

        self.config = DockerfileConfig(
            base_image=base.get("baseImage", "nginx:alpine"),
            expose_port=int(base.get("exposePort", 80)),
            copy_source=base.get("copySource", "./html"),
            work_dir=base.get("workDir", "/usr/share/nginx/html"),
            groq_api_endpoint=base.get("groqApiEndpoint", self.config.groq_api_endpoint),
            groq_api_key=base.get("groqApiKey", self.config.groq_api_key),
        )

    def generate_dockerfile(self) -> str:
        """
        Generate Dockerfile content based on the current configuration.
        
        Returns:
            str: Complete Dockerfile content with appropriate instructions
                for building a container image
        """
        dockerfile = f"""
FROM {self.config.base_image}

WORKDIR {self.config.work_dir}

COPY {self.config.copy_source} .

EXPOSE {self.config.expose_port}

CMD ["nginx", "-g", "daemon off;"]
"""
        # Ensure FROM is on the first line (no leading newline)
        return dockerfile.lstrip()