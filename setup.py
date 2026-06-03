from pathlib import Path
from typing import List, Set

from setuptools import setup, find_packages

BASE_DIR = Path(__file__).resolve().parent


def read_requirements(rel_path: str) -> List[str]:
    lines = (BASE_DIR / rel_path).read_text().splitlines()
    reqs: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        reqs.append(stripped)
    return reqs


def dedupe(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


REQUIREMENTS = dedupe(
    read_requirements("requirements.txt")
)

setup(
    name="GN_Bench",
    version="0.1.0",
    description="GN_Bench_Tools: a suite for embodied agent tasks and benchmarks",
    author="Yuehao Huang",
    author_email="yuehaohuang@zju.edu.cn",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
)
