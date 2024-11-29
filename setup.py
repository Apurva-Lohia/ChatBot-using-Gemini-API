from setuptools import find_packages,setup

setup(
    name='ChatBot using Gemini API',
    version='0.0.1',
    author='apurva lohia',
    author_email='lohiaapurva@gmail.com',
    install_requires=["google-generativeai","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages(where="src"),  # Look for packages in src directory
    package_dir={"": "src"},  # Root package is in src
)