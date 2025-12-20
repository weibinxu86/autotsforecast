from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='autotsforecast',
    version='0.1.0',
    author='Weibin Xu',
    author_email='weibinxu86@gmail.com',
    description='Automated multivariate time series forecasting with model selection, backtesting, hierarchical reconciliation, and SHAP interpretability',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/weibinxu86/autotsforecast',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'statsmodels>=0.13.0',
        'scipy>=1.7.0',
        'joblib>=1.1.0',
    ],
    extras_require={
        'all': [
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
            'shap>=0.42.0',
            'xgboost>=1.5.0',
        ],
        'viz': [
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
        ],
        'interpret': [
            'shap>=0.42.0',
        ],
        'ml': [
            'xgboost>=1.5.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)