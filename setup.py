from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='langchain-llm7',
    version='2025.10.271728',
    author='Eugene Evstafev',
    author_email='hi@eugene.plus',
    description='A LangChain wrapper for LLM7 API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chigwell/langchain_llm7',
    packages=find_packages(),
    install_requires=[
        'tokeniser>=0.0.3',
        'requests==2.32.5',
        'pydantic==2.11.4',
        'langchain-core==1.0.1'
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    license='Apache-2.0',
    tests_require=['unittest'],
    test_suite='test',
)