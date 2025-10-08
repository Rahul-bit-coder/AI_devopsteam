from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Optional
import os
import subprocess


class KubernetesConfig(BaseModel):
    """
    Configuration for the Kubernetes agent.

    Attributes:
        app_name: Logical application name used in metadata and labels
        image: Container image (e.g., myapp:latest or ghcr.io/user/app:tag)
        replicas: Number of desired replicas for the Deployment
        container_port: Port exposed by the container
        service_type: Service type (ClusterIP, NodePort, LoadBalancer)
        namespace: Target namespace (created if missing when applying)
        manifests_dir: Directory to write manifests to
    """
    app_name: str = "myapp"
    image: str = "myapp:latest"
    replicas: int = 1
    container_port: int = 80
    # Externally exposed service port (ClusterIP/NodePort/LoadBalancer service's port)
    service_port: int = 80
    service_type: str = "ClusterIP"
    namespace: str = "default"
    manifests_dir: str = "k8s"


class KubernetesAgent(Agent):
    config: KubernetesConfig

    def __init__(self, config: KubernetesConfig):
        super().__init__()
        self.config = config

    def _ensure_dir(self, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    def generate_manifests(self) -> dict:
        """
        Generate Deployment and Service YAML manifests and save them to disk.
        Returns a dict with written file paths.
        """
        self._ensure_dir(self.config.manifests_dir)

        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.config.app_name}
  labels:
    app: {self.config.app_name}
spec:
  replicas: {self.config.replicas}
  selector:
    matchLabels:
      app: {self.config.app_name}
  template:
    metadata:
      labels:
        app: {self.config.app_name}
    spec:
      containers:
      - name: {self.config.app_name}
        image: {self.config.image}
        ports:
        - containerPort: {self.config.container_port}
""".lstrip()

        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: {self.config.app_name}
  labels:
    app: {self.config.app_name}
spec:
  selector:
    app: {self.config.app_name}
  ports:
  - port: {self.config.service_port}
    targetPort: {self.config.container_port}
  type: {self.config.service_type}
""".lstrip()

        deployment_path = os.path.join(self.config.manifests_dir, "deployment.yaml")
        service_path = os.path.join(self.config.manifests_dir, "service.yaml")

        with open(deployment_path, "w", encoding="utf-8") as f:
            f.write(deployment_yaml)
        with open(service_path, "w", encoding="utf-8") as f:
            f.write(service_yaml)

        return {"deployment": deployment_path, "service": service_path}

    def apply_manifests(self) -> str:
        """
        Apply manifests using kubectl. Returns a status string.
        """
        try:
            # Optionally create namespace if not default
            if self.config.namespace and self.config.namespace != "default":
                subprocess.run(
                    ["kubectl", "create", "ns", self.config.namespace],
                    capture_output=True, text=True
                )

            result = subprocess.run(
                [
                    "kubectl", "apply", "-n", self.config.namespace,
                    "-f", os.path.join(self.config.manifests_dir, "deployment.yaml"),
                    "-f", os.path.join(self.config.manifests_dir, "service.yaml"),
                ],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                return f"kubectl apply failed: {result.stderr.strip()}"
            return f"kubectl apply succeeded:\n{result.stdout.strip()}"
        except FileNotFoundError:
            return "kubectl is not installed or not available in PATH."
        except Exception as e:
            return f"Error applying manifests: {str(e)}"


