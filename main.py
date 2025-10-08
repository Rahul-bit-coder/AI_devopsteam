from agents.github_actions_agent import GitHubActionsAgent, GitHubActionsConfig
from agents.dockerfile_agent import DockerfileAgent, DockerfileConfig
from agents.build_predictor_agent import BuildPredictorAgent, BuildPredictorConfig
from agents.build_status_agent import BuildStatusAgent, BuildStatusConfig
from agents.kubernetes_agent import KubernetesAgent, KubernetesConfig
from agents.code_review_agent import CodeReviewAgent, CodeReviewConfig
import os
from dotenv import load_dotenv
from typing import Optional
import re

# Load environment variables from .env file
load_dotenv()

def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists. If it doesn't, create it (including parents).
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_github_actions_pipeline() -> str:
    """
    Create a GitHub Actions workflow file using the GitHubActionsAgent.
    Returns the path to the created workflow file.
    """
    print("\n1️⃣ GitHub Actions Agent: Creating CI/CD Pipeline...")
    gha_config = GitHubActionsConfig(
        workflow_name="CI Pipeline",
        python_version="3.13.0",
        run_tests=True,
        groq_api_endpoint=os.getenv("GROQ_API_ENDPOINT"),
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    gha_agent = GitHubActionsAgent(config=gha_config)
    pipeline = gha_agent.generate_pipeline()

    workflows_dir = ".github/workflows"
    ensure_dir(workflows_dir)
    workflow_path = os.path.join(workflows_dir, "CI3.yml")
    with open(workflow_path, "w", encoding="utf-8") as f:
        f.write(pipeline)
    print("✅ CI/CD Pipeline created!")
    # Lightweight validation: basic structure checks
    if ("name:" in pipeline) and ("jobs:" in pipeline):
        print("🔎 Workflow validation: basic structure OK")
    else:
        print("⚠️ Workflow validation: missing expected keys ('name', 'jobs')")
    return workflow_path


def create_dockerfile() -> str:
    """
    Create a Dockerfile using the DockerfileAgent. Returns the path to the Dockerfile.
    """
    print("\n2️⃣ Dockerfile Agent: Creating Dockerfile...")
    docker_config = DockerfileConfig(
        base_image="nginx:alpine",
        expose_port=80,
        copy_source="./html",
        work_dir="/usr/share/nginx/html",
        groq_api_endpoint=os.getenv("GROQ_API_ENDPOINT"),
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    docker_agent = DockerfileAgent(config=docker_config)
    dockerfile = docker_agent.generate_dockerfile()

    dockerfile_path = "Dockerfile"
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile)
    print("✅ Dockerfile created!")
    # Lightweight validation: ensure critical directives exist
    if ("FROM" in dockerfile) and ("CMD" in dockerfile):
        print("🔎 Dockerfile validation: basic directives OK")
    else:
        print("⚠️ Dockerfile validation: expected directives missing")
    return dockerfile_path


def build_and_check_status(image_tag: str = "myapp:latest") -> str:
    """
    Build the Docker image and check its status using the BuildStatusAgent.
    Returns a human-readable status string.
    """
    print("\n3️⃣ Build Status Agent: Building and checking Docker image...")
    status_config = BuildStatusConfig(image_tag=image_tag)
    status_agent = BuildStatusAgent(config=status_config)

    print("🔨 Building Docker image...")
    import subprocess
    try:
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            capture_output=True,
            text=True
        )
        if build_result.returncode != 0:
            print("❌ Docker build failed. Showing stderr:")
            print(build_result.stderr)
        else:
            print("✅ Docker build completed.")
    except FileNotFoundError:
        return "Docker is not installed or not available in PATH."

    status = status_agent.check_build_status()
    print(f"📊 Build Status: {status}")
    return status


def predict_build(build_status_message: str) -> dict:
    """
    Run the BuildPredictorAgent to predict build success/failure.
    Returns the prediction dict.
    """
    print("\n4️⃣ Build Predictor Agent: Analyzing build patterns...")
    predictor_config = BuildPredictorConfig(
        model="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    predictor_agent = BuildPredictorAgent(config=predictor_config)
    build_data = {
        "dockerfile_exists": os.path.isfile("Dockerfile"),
        "ci_pipeline_exists": os.path.isfile(os.path.join(".github", "workflows", "CI3.yml")),
        "last_build_status": build_status_message,
        "python_version": "3.13.0",
        "dependencies_updated": True,
    }
    prediction = predictor_agent.predict_build_failure(build_data)
    print(f"🔮 Build Prediction: {prediction}")
    return prediction


def _extract_overrides(prompt: str) -> dict:
    """
    Extract simple overrides from the prompt, like replicas and ports.
    Supports phrases like:
      - replicas 3 / 3 replicas
      - expose 81 / port 81 / service port 81
      - container port 8080 / target port 8080
      - service type loadbalancer/nodeport/clusterip
    """
    text = (prompt or "").lower()
    overrides: dict = {}

    # replicas
    m = re.search(r"replicas?\s*(?:to\s*)?(\d+)", text)
    if m:
        overrides["replicas"] = int(m.group(1))

    # service port (external)
    m = re.search(r"(?:expose|service\s*port|port)\s*(\d{2,5})", text)
    if m:
        overrides["service_port"] = int(m.group(1))

    # container port (targetPort)
    m = re.search(r"(?:container\s*port|target\s*port)\s*(\d{2,5})", text)
    if m:
        overrides["container_port"] = int(m.group(1))

    # service type
    if "loadbalancer" in text:
        overrides["service_type"] = "LoadBalancer"
    elif "nodeport" in text:
        overrides["service_type"] = "NodePort"
    elif "clusterip" in text:
        overrides["service_type"] = "ClusterIP"

    return overrides


def handle_prompt(prompt: str) -> None:
    """
    Route a natural-language prompt to an appropriate agent task.
    Supported intents: pipeline, dockerfile, build, predict, k8s, kubernetes, deploy, review, validate, all.
    """
    normalized = (prompt or "").strip().lower()
    if not normalized:
        print("No task provided. Try: 'create pipeline', 'generate dockerfile', 'build', 'predict', or 'all'.")
        return

    # Simple keyword-based intent routing
    if any(k in normalized for k in ["all", "everything", "run all"]):
        create_github_actions_pipeline()
        create_dockerfile()
        status = build_and_check_status()
        predict_build(status)
        # Generate k8s manifests but don't force apply
        k8s_generate_and_optionally_apply(apply=False, overrides=_extract_overrides(prompt))
        print("\n✨ DevOps AI Team has completed their tasks!")
        return

    if "pipeline" in normalized or "github actions" in normalized or "workflow" in normalized:
        create_github_actions_pipeline()
        return

    if "dockerfile" in normalized:
        create_dockerfile()
        return

    if "build" in normalized:
        build_and_check_status()
        return

    if "predict" in normalized or "prediction" in normalized:
        # Best effort: run prediction using current known status
        status = "Unknown"
        try:
            status = build_and_check_status()
        except Exception:
            pass
        predict_build(status)
        return

    if any(k in normalized for k in ["k8s", "kubernetes", "deploy", "manifest", "manifests"]):
        # Detect if user wants to apply
        apply = any(k in normalized for k in ["apply", "kubectl apply", "deploy now"]) 
        k8s_generate_and_optionally_apply(apply=apply, overrides=_extract_overrides(prompt))
        return

    if any(k in normalized for k in ["review", "code review", "lint", "analyze code"]):
        # Extract a simple file path token if provided: e.g. "review html/talkitdoit.html"
        tokens = normalized.split()
        target_path = None
        for t in tokens:
            if t.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css')):
                target_path = t
                break
        review_local(target_path or "html/talkitdoit.html")
        return

    if "validate" in normalized:
        # Run lightweight validations; optional docker build check
        try:
            k8s_generate_and_optionally_apply(apply=False)
        except Exception:
            pass
        # Try a quick docker build to validate Dockerfile if present
        if os.path.isfile("Dockerfile"):
            print("\n🧪 Validating Dockerfile by building image (no cache)...")
            import subprocess
            result = subprocess.run(["docker", "build", "--no-cache", "-t", "myapp:latest", "."], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker build validation succeeded")
            else:
                print("❌ Docker build validation failed\n" + result.stderr)
        return

    print("Unrecognized task. Try: 'create pipeline', 'generate dockerfile', 'build', 'predict', or 'all'.")


def k8s_generate_and_optionally_apply(apply: bool = False, overrides: Optional[dict] = None) -> None:
    print("\n5️⃣ Kubernetes Agent: Generating manifests...")
    image_tag = "myapp:latest"
    overrides = overrides or {}
    kcfg = KubernetesConfig(
        app_name="myapp",
        image=image_tag,
        replicas=overrides.get("replicas", 1),
        container_port=overrides.get("container_port", 80),
        service_port=overrides.get("service_port", overrides.get("container_port", 80)),
        service_type=overrides.get("service_type", "ClusterIP"),
        namespace="default",
        manifests_dir="k8s",
    )
    kagent = KubernetesAgent(config=kcfg)
    files = kagent.generate_manifests()
    print(f"✅ Kubernetes manifests created: {files}")
    if apply:
        print("🚀 Applying manifests to cluster...")
        result = kagent.apply_manifests()
        print(result)


def review_local(file_path: str) -> None:
    print("\n6️⃣ Code Review Agent: Reviewing local file...")
    cfg = CodeReviewConfig(
        model="llama3-8b-8192",
        groq_api_endpoint=os.getenv("GROQ_API_ENDPOINT", "https://api.groq.com/v1"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        github_token=os.getenv("GH_TOKEN", ""),
        repo_name="",
        pull_request_number=0,
    )
    agent = CodeReviewAgent(config=cfg)
    result = agent.review_local_file(file_path)
    print(result)


def main(task_prompt: Optional[str] = None):
    """
    Main orchestration function that coordinates the DevOps AI team's activities.
    
    This function manages four main tasks:
    1. Creating a GitHub Actions CI/CD pipeline
    2. Generating a Dockerfile
    3. Building and checking Docker image status
    4. Predicting build success/failure
    """
    print("🤖 DevOps AI Team Starting Up...")
    if task_prompt is None:
        try:
            task_prompt = input(
                "\nWhat would you like me to do? (e.g., 'create pipeline', 'generate dockerfile', 'build', 'predict', 'all')\n> "
            )
        except EOFError:
            task_prompt = "all"

    handle_prompt(task_prompt)

if __name__ == "__main__":
    # Support optional command-line arg as the task prompt
    import sys
    cli_prompt = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_prompt)