from setuptools import setup, find_packages

setup(
    name="super_resolution_chaldene",  # Change to your package name
    version="0.1.0",
    author="Fei Chen",
    author_email="boxchenfei@gmail.com",
    description="Super resolution based on EMDiffuse",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Max-ChenFei/super_resolution",  # Change to your GitHub repo
    packages=find_packages(),
    install_requires=[],  # Add dependencies if needed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)