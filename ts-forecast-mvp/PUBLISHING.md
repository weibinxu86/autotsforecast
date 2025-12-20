# Publishing to PyPI

## Preparation Checklist

Before publishing to PyPI, ensure:

1. ✅ All tests pass: `pytest tests/`
2. ✅ Version number updated in `setup.py` and `src/autotsforecast/__init__.py`
3. ✅ CHANGELOG.md updated with release notes
4. ✅ README.md is complete and accurate
5. ✅ LICENSE file is present (MIT)
6. ✅ All examples work correctly

## Installation Requirements

```bash
pip install build twine
```

## Build Distribution Packages

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
python -m build
```

This creates:
- `dist/autotsforecast-X.Y.Z.tar.gz` (source distribution)
- `dist/autotsforecast-X.Y.Z-py3-none-any.whl` (wheel distribution)

## Test on TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple autotsforecast
```

## Publish to PyPI

```bash
# Upload to PyPI (production)
python -m twine upload dist/*
```

You'll be prompted for your PyPI credentials. Alternatively, configure API token in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
```

## After Publishing

Users can install via:

```bash
# Basic installation
pip install autotsforecast

# With all optional dependencies
pip install autotsforecast[all]

# With specific features
pip install autotsforecast[viz,interpret,ml]
```

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## Updating a Release

1. Increment version number
2. Update CHANGELOG.md
3. Rebuild: `python -m build`
4. Upload: `python -m twine upload dist/*`

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Packaging Guide: https://packaging.python.org/tutorials/packaging-projects/
