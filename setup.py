from setuptools import setup, find_packages

setup(
  name = 'retro-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.3.8',
  license='MIT',
  description = 'RETRO - Retrieval Enhanced Transformer - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/RETRO-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention-mechanism',
    'retrieval',
  ],
  install_requires=[
    'transformers',
    'einops>=0.3',
    'numpy',
    'sentencepiece',
    'torch>=1.6',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
