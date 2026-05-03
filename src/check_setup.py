import importlib
import os
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "pandas",
    "akshare",
    "baostock",
    "matplotlib",
    "sklearn",
    "joblib",
]


def configure_local_cache() -> None:
    cache_dir = Path(".matplotlib_cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def is_package_available(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def main() -> None:
    configure_local_cache()

    print("QuantPilot-AI Setup Check")
    print("-------------------------")

    missing_packages = []
    for package_name in REQUIRED_PACKAGES:
        if is_package_available(package_name):
            print(f"{package_name}: OK")
        else:
            print(f"{package_name}: MISSING")
            missing_packages.append(package_name)

    print()
    if missing_packages:
        print("Setup check failed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print("Setup check passed.")


if __name__ == "__main__":
    main()
