from setuptools import find_packages
from setuptools import setup
import os

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="llm_selenium_agent",
    version="0.0.1",
    author="Lunde Chen",
    author_email="lundechen@shu.edu.cn",
    maintainer="Lunde Chen",
    maintainer_email="lundechen@shu.edu.cn",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    scripts=[
        "scripts/llm_selenium_agent_first_time_setup",
        "scripts/llm_selenium_agent_streamlit_app",
    ],
    package_data={
        "llm_selenium_agent": [
            "streamlit_app/css/*.css",
            "Readme.md",
        ],
    },
    include_package_data=True,
)
