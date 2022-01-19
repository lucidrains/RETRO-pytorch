from setuptools import setup, find_packages

setup(
  name = 'retro-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.12',
  license='MIT',
  description = 'RETRO - Retrieval Enhanced Transformer - Pytorch',
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
    'einops>=0.3',
    'faiss-gpu',
    'numpy',
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
