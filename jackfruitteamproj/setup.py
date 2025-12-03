"""
setup.py
Setup script for Python 3.14
"""

from setuptools import setup, find_packages

setup(
    name='quantum_trading_system',
    version='1.0.0',
    description='Quantum-Inspired Financial Trading System',
    author='Trading Team',
    python_requires='>=3.14',
    packages=find_packages(),
    install_requires=[
        'pennylane>=0.38.0',
        'qiskit>=1.3.0',
        'numpy>=2.1.0',
        'pandas>=2.2.0',
        'scipy>=1.14.0',
        'scikit-learn>=1.5.0',
        'tensorflow>=2.18.0',
        'transformers>=4.46.0',
        'yfinance>=0.2.49',
        'ccxt>=4.4.0',
        'plotly>=5.24.0',
        'streamlit>=1.40.0',
        'aiohttp>=3.11.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.14',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
    ],
)