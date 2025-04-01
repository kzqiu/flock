from setuptools import setup, find_packages

setup(
    name="flock-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym>=0.17.0",
        "pygame>=2.0.0",
        "shimmy",
        "stable-baselines3[extra]",
        "torch"
    ],
    author="Anupam Bhakta, Ben Fu, Kevin Qiu",
    author_email="ab5494@columbia.edu, bhf2117@columbia.edu, kzq2000@columbia.edu",
    description="A reinforcement learning environment for cooperative transport",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kzqiu/flock",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)