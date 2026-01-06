# Publishing autotsforecast to PyPI

This guide walks you through publishing your package to PyPI so users can install it via `pip install autotsforecast`.

## Prerequisites

1. **PyPI Account**
   - Create account at: https://pypi.org/account/register/
   - Verify your email address

2. **TestPyPI Account (Optional but Recommended)**
   - Create account at: https://test.pypi.org/account/register/
   - Use this for testing before publishing to production PyPI

3. **API Tokens (Recommended over passwords)**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token (save it securely - shown only once)
   - Repeat for TestPyPI: https://test.pypi.org/manage/account/token/

## Step 1: Verify Your Package is Ready

```powershell
# From the project root (c:\forecasting\autotsforecast)
cd c:\forecasting\autotsforecast

# Ensure build dependencies are installed
pip install --upgrade build twine

# Clean previous builds
Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue

# Build fresh distribution packages
python -m build
```

**Expected output:**
```
dist/
  autotsforecast-0.1.0-py3-none-any.whl
  autotsforecast-0.1.0.tar.gz
```

## Step 2: Verify Package Contents (Optional)

```powershell
# Check wheel contents
python -m zipfile -l dist/autotsforecast-0.1.0-py3-none-any.whl

# Check sdist contents
tar -tzf dist/autotsforecast-0.1.0.tar.gz
```

Ensure:
- Source code is included (`src/autotsforecast/`)
- Documentation files are present (README.md, LICENSE, etc.)
- No unwanted files (tests, examples, __pycache__, etc.)

## Step 3: Test Upload to TestPyPI (Recommended)

TestPyPI is a separate instance of PyPI for testing. Always test here first!

```powershell
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

When prompted:
- **Username:** `__token__`
- **Password:** Your TestPyPI API token (starts with `pypi-...`)

**Verify the upload:**
- Visit: https://test.pypi.org/project/autotsforecast/
- Check that all metadata displays correctly
- Verify README renders properly

## Step 4: Test Installation from TestPyPI

Create a fresh virtual environment and test install:

```powershell
# Create test environment
python -m venv test_env
test_env\Scripts\Activate.ps1

# Install from TestPyPI (note the extra index for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ autotsforecast[all]

# Test basic import
python -c "import autotsforecast; print(autotsforecast.__version__)"

# Try creating a model
python -c "from autotsforecast import VARForecaster; print('Import successful!')"

# Deactivate and remove test env
deactivate
Remove-Item -Recurse -Force test_env
```

If everything works, proceed to production PyPI!

## Step 5: Upload to Production PyPI

⚠️ **IMPORTANT:** Once uploaded to PyPI, you **cannot** delete or re-upload the same version number. Make sure everything is correct!

```powershell
# Upload to production PyPI
python -m twine upload dist/*
```

When prompted:
- **Username:** `__token__`
- **Password:** Your PyPI API token (starts with `pypi-...`)

## Step 6: Verify Production Upload

1. **Check PyPI page:**
   - Visit: https://pypi.org/project/autotsforecast/
   - Verify metadata, description, and links

2. **Test installation:**
   ```powershell
   # Create fresh environment
   python -m venv verify_env
   verify_env\Scripts\Activate.ps1
   
   # Install from PyPI
   pip install autotsforecast[all]
   
   # Quick test
   python -c "from autotsforecast import AutoForecaster; print('Success!')"
   
   # Cleanup
   deactivate
   Remove-Item -Recurse -Force verify_env
   ```

## Step 7: Announce Your Release

1. **GitHub Release:**
   - Create a release tag: `v0.1.0`
   - Add release notes from CHANGELOG.md
   - Link to PyPI page

2. **Update README badges (optional):**
   ```markdown
   [![PyPI version](https://badge.fury.io/py/autotsforecast.svg)](https://pypi.org/project/autotsforecast/)
   [![Downloads](https://pepy.tech/badge/autotsforecast)](https://pepy.tech/project/autotsforecast)
   ```

## Common Issues & Solutions

### Issue: "File already exists"
**Solution:** You tried to upload the same version twice. Increment version in `pyproject.toml` and rebuild.

### Issue: "Invalid or non-rendering README"
**Solution:** 
- Ensure `README.md` exists in project root
- Verify `readme = "README.md"` in `pyproject.toml`
- Check for syntax errors in markdown

### Issue: "Missing dependencies"
**Solution:** 
- Verify all dependencies listed in `pyproject.toml` under `[project.dependencies]`
- Test in fresh environment first

### Issue: Authentication failed
**Solution:**
- Username must be `__token__` (not your PyPI username)
- Password is your API token (not account password)
- Token must have upload permissions

## Updating Your Package (Future Releases)

When releasing a new version:

1. **Update version number:**
   ```toml
   # pyproject.toml
   version = "0.1.1"  # or 0.2.0, 1.0.0, etc.
   ```

2. **Update CHANGELOG.md** with new features/fixes

3. **Clean and rebuild:**
   ```powershell
   Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
   python -m build
   ```

4. **Upload new version:**
   ```powershell
   python -m twine upload dist/*
   ```

## Using GitHub Actions for Automated Publishing (Advanced)

Create `.github/workflows/publish.yml` for automatic publishing on release:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  pypi-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: python -m pip install --upgrade build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```

Add your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`.

## Resources

- **PyPI Packaging Guide:** https://packaging.python.org/tutorials/packaging-projects/
- **Twine Documentation:** https://twine.readthedocs.io/
- **Semantic Versioning:** https://semver.org/
- **Python Packaging Authority:** https://www.pypa.io/

## Quick Reference Commands

```powershell
# Clean build
Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue

# Build
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*

# Check built packages
python -m twine check dist/*
```

---

**You're all set!** Users can now install your package with:
```bash
pip install autotsforecast[all]
```
