import importlib


def available_packages() -> dict[str, str]:
    all_available_packages = {}
    for dist in importlib.metadata.distributions():
        package_name = dist.metadata["Name"]
        package_version = dist.version
        all_available_packages[package_name] = package_version
    return all_available_packages
