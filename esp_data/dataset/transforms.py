"""Module for defining and applying data transformation pipelines

Example:
# Define a pipeline
transforms = [
    {
        "module_path": "myproject.transforms.normalize",
        "function_name": "normalize_data",
        "parameters": {"mean": 0, "std": 1}
    },
    {
        "module_path": "myproject.transforms.augment",
        "function_name": "add_noise",
        "parameters": {"noise_level": 0.1}
    }
]

pipeline = create_pipeline(
    transforms=transforms,
    name="normalize_and_augment",
    description="Normalizes data and adds noise",
    git_repo_path="/path/to/repo"
)

# Save pipeline configuration
pipeline.save("pipeline_config.json")

# Load and use pipeline
loaded_pipeline = TransformPipeline.load("pipeline_config.json")
transformed_data = loaded_pipeline.apply(data)

# The saved JSON will look something like this:
{
  "name": "normalize_and_augment",
  "description": "Normalizes data and adds noise",
  "created_at": "2025-02-04T10:30:00",
  "git_commit": "abc123...",
  "git_repo_path": "/path/to/repo",
  "steps": [
    {
      "module_path": "myproject.transforms.normalize",
      "function_name": "normalize_data",
      "parameters": {"mean": 0, "std": 1},
      "version": "1.2.3"
    },
    {
      "module_path": "myproject.transforms.augment",
      "function_name": "add_noise",
      "parameters": {"noise_level": 0.1},
      "version": "0.5.0"
    }
  ]
}

"""

import importlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import git
import pkg_resources

from esp_data.utils import make_simple_logger

logger = make_simple_logger("esp_data.transforms")


@dataclass
class TransformStep:
    module_path: str
    function_name: str
    parameters: Dict[str, Any]
    version: str = None  # Package version if available

    def __post_init__(self):
        if not self.version:
            try:
                # Try to get package version
                module_name = self.module_path.split(".")[0]
                self.version = pkg_resources.get_distribution(module_name).version
            except Exception as e:
                logger.warning(f"Could not get version for {module_name}: {e}")
                self.version = "unknown"

    def get_transform_function(self):
        module = importlib.import_module(self.module_path)
        return getattr(module, self.function_name)


@dataclass
class TransformPipeline:
    steps: List[TransformStep]
    name: str
    git_commit: Optional[str] = None
    git_repo_path: Optional[str] = None
    created_at: str = None
    description: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.git_repo_path:
            self.update_git_info()

    def update_git_info(self):
        try:
            repo = git.Repo(self.git_repo_path)
            self.git_commit = repo.head.commit.hexsha
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
            self.git_commit = None

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=lambda x: x.__dict__)

    @classmethod
    def load(cls, path: str) -> "TransformPipeline":
        with open(path) as f:
            data = json.load(f)
            # Convert steps back to TransformStep objects
            data["steps"] = [TransformStep(**step) for step in data["steps"]]
            return cls(**data)

    def apply(self, data: Any) -> Any:
        """Apply all transformation steps in sequence"""
        result = data
        for step in self.steps:
            transform_fn = step.get_transform_function()
            result = transform_fn(result, **step.parameters)
        return result


def create_pipeline(
    transforms: List[Union[dict, TransformStep]],
    name: str,
    description: Optional[str] = None,
    git_repo_path: Optional[str] = None,
) -> TransformPipeline:
    """Helper function to create a pipeline from a list of transform configurations"""
    steps = []
    for t in transforms:
        if isinstance(t, dict):
            steps.append(TransformStep(**t))
        else:
            steps.append(t)

    return TransformPipeline(steps=steps, name=name, description=description, git_repo_path=git_repo_path)
