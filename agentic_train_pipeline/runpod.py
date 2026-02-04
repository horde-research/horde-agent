"""RunPod setup helpers and strict Pydantic schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunPodInstanceConfig(BaseModel):
    """Configuration details for a RunPod instance."""

    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(..., description="Cloud provider name, e.g. runpod")
    gpu_type: str = Field(..., description="GPU type, e.g. A10, A100, L4")
    gpu_count: int = Field(1, ge=1, description="Number of GPUs")
    cpu_cores: int = Field(8, ge=2, description="CPU cores")
    ram_gb: int = Field(32, ge=8, description="System RAM in GB")
    storage_gb: int = Field(80, ge=20, description="Disk storage in GB")
    python_version: str = Field("3.10", description="Python version for venv/conda")
    cuda_version: Optional[str] = Field(
        None, description="CUDA version, must match installed torch build"
    )


class ProjectDependencies(BaseModel):
    """Pinned (or loosely pinned) dependencies required for the project."""

    model_config = ConfigDict(extra="forbid", strict=True)

    python_requires: str = Field(">=3.10", description="Python version constraint")
    pip_packages: List[str] = Field(
        default_factory=lambda: [
            "openai>=1.30.0",
            "pydantic>=2.6.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "peft>=0.10.0",
            "accelerate>=0.30.0",
            "torch>=2.1.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "rich>=13.7.0",
        ]
    )


class RunPodSetupPlan(BaseModel):
    """End-to-end setup plan for RunPod."""

    model_config = ConfigDict(extra="forbid", strict=True)

    instance: RunPodInstanceConfig
    dependencies: ProjectDependencies
    repo_path: str = Field("/workspace/agentic-train-pipeline", description="Repo path")

    def pip_install_command(self) -> str:
        packages = " ".join(self.dependencies.pip_packages)
        return f"pip install -U pip && pip install -e . {packages}"

