"""
Module for defining and applying data transformation pipelines
"""

import importlib
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import git
import pkg_resources

from esp_data.io import AnyPath, anypath
from esp_data.utils import utc_now_str

logger = logging.getLogger("esp_data")


@dataclass
class SimpleTransformStep:
    """A simplified transform step that works with a function reference"""

    function: Callable
    parameters: Dict[str, Any] = None
    name: str = None
    module_path: str = None
    function_name: str = None
    version: str = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.name is None:
            self.name = self.function.__name__

        # Get module path and function name for serialization
        if self.module_path is None:
            module = inspect.getmodule(self.function)
            self.module_path = module.__name__ if module else "unknown"

        if self.function_name is None:
            self.function_name = self.function.__name__

        # Get package version
        if self.version is None:
            try:
                # Extract the top-level package name
                top_package = self.module_path.split(".")[0]
                self.version = pkg_resources.get_distribution(top_package).version
            except Exception:
                self.version = "unknown"

    def __call__(self, *args):
        """Apply the transform function to the data"""
        return self.function(*args, **self.parameters)

    def to_dict(self):
        """Convert to a serializable dictionary"""
        return {
            "name": self.name,
            "module_path": self.module_path,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a step from a dictionary"""
        try:
            module = importlib.import_module(data["module_path"])
            function = getattr(module, data["function_name"])

            return cls(
                function=function,
                parameters=data["parameters"],
                name=data["name"],
                module_path=data["module_path"],
                function_name=data["function_name"],
                version=data["version"],
            )
        except ImportError:
            raise ImportError(f"Could not import module {data['module_path']}")
        except AttributeError:
            raise AttributeError(f"Could not find function {data['function_name']} in module {data['module_path']}")


@dataclass
class TransformStep:
    """A transform step that can be a function or a class method"""

    module_path: str
    function_name: str
    parameters: Dict[str, Any]
    version: str = None  # Package version if available
    is_class_method: bool = False  # Flag to indicate if it's a class method
    class_name: Optional[str] = None  # Class name if it's a class method
    init_parameters: Optional[Dict[str, Any]] = None  # Parameters for class initialization

    def __post_init__(self):
        if not self.version:
            try:
                # Try to get package version
                module_name = self.module_path.split(".")[0]
                self.version = pkg_resources.get_distribution(module_name).version
            except Exception as e:
                logger.warning(f"Could not get version for {module_name}: {e}")
                self.version = "unknown"

        # Initialize default parameters
        if self.is_class_method and self.init_parameters is None:
            self.init_parameters = {}

    def get_transform_function(self):
        """
        Get the transform function or method.

        For regular functions, returns the function directly.
        For class methods, returns a wrapper function that creates an instance
        and calls the method.
        """
        module = importlib.import_module(self.module_path)

        if not self.is_class_method:
            # Regular function case - same as before
            return getattr(module, self.function_name)
        else:
            # Class method case
            if not self.class_name:
                raise ValueError("class_name must be provided for class methods")

            # Get the class
            class_obj = getattr(module, self.class_name)

            # Create a wrapper function that:
            # 1. Instantiates the class with init_parameters
            # 2. Calls the specified method with parameters
            def class_method_wrapper(data, **kwargs):
                # Merge default parameters with any provided at call time
                merged_params = {**self.parameters, **kwargs}

                # Create an instance of the class
                instance = class_obj(**self.init_parameters)

                # Get the method from the instance
                method = getattr(instance, self.function_name)

                # Call the method with the data and parameters
                return method(data, **merged_params)

            return class_method_wrapper

    def to_dict(self):
        """Convert to a serializable dictionary"""
        return {
            "module_path": self.module_path,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "version": self.version,
            "is_class_method": self.is_class_method,
            "class_name": self.class_name,
            "init_parameters": self.init_parameters,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a step from a dictionary"""
        return cls(
            module_path=data["module_path"],
            function_name=data["function_name"],
            parameters=data["parameters"],
            version=data.get("version", None),
            is_class_method=data.get("is_class_method", False),
            class_name=data.get("class_name", None),
            init_parameters=data.get("init_parameters", None),
        )

    def __call__(self, *data):
        """Apply the transform function to the data"""
        transform_fn = self.get_transform_function()
        return transform_fn(*data, **self.parameters)


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
            self.created_at = utc_now_str()
        if self.git_repo_path:
            self.update_git_info()

    def update_git_info(self):
        try:
            repo = git.Repo(self.git_repo_path)
            self.git_commit = repo.head.commit.hexsha
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
            self.git_commit = None

    def to_dict(self):
        """Convert to a serializable dictionary"""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "name": self.name,
            "git_commit": self.git_commit,
            "git_repo_path": self.git_repo_path,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a pipeline from a dictionary"""
        steps = [TransformStep.from_dict(step) for step in data["steps"]]
        return cls(steps=steps, name=data["name"], git_repo_path=data["git_repo_path"])

    def save(self, path: str | AnyPath) -> None:
        with anypath(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | AnyPath) -> "TransformPipeline":
        with anypath(path).open("r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __call__(self, *data) -> Any:
        """Apply all transformation steps in sequence"""
        result = data
        for step in self.steps:
            transform_fn = step.get_transform_function()
            if isinstance(result, tuple):
                result = transform_fn(*result, **step.parameters)
            else:
                result = transform_fn(result, **step.parameters)

        return result


### TEST FUNCTIONS ###
def test_method_add_int(a: int, b: int) -> int:
    return a + b


def test_method_multiply(a: int, b: int) -> int:
    return a * b


class TestClass:
    def __init__(self, factor: int):
        self.factor = factor

    def multiply(self, a: int) -> int:
        return a * self.factor
