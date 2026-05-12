"""Utility functions for SWE-Bench / SWE-Gym Docker image naming."""

import os


# Override the default Docker registry via env var.
# e.g. SWE_DOCKER_REGISTRY=slime-agent-cn-beijing.cr.volces.com
_DEFAULT_REGISTRY = "docker.io"


def get_docker_image_name(instance: dict, data_source: str) -> str:
    """Get the image name for a SWEBench/SWE-Gym instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        registry = os.getenv("SWE_DOCKER_REGISTRY", _DEFAULT_REGISTRY)
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")
            image_name = f"{registry}/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"{registry}/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    return image_name
