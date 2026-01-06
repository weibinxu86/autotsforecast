# Quick Publishing Commands

## First Time Setup (One Time Only)

1. **Install publishing tools:**
   ```powershell
   pip install --upgrade build twine
   ```

2. **Get PyPI API token:**
   - Go to: https://pypi.org/account/register/ (create account if needed)
   - Go to: https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: "autotsforecast-upload"
   - Scope: "Entire account" (or specific project after first upload)
   - **SAVE THE TOKEN** - you'll only see it once!

## Publishing Steps (Run These Now)

```powershell
# 1. Navigate to project
cd c:\forecasting\autotsforecast

# 2. Clean previous builds
Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue

# 3. Build the package
python -m build

# 4. Check the build (optional but recommended)
python -m twine check dist/*

# 5. Upload to PyPI
python -m twine upload dist/*
```

When prompted by step 5:
- **Username:** `__token__`
- **Password:** Paste your PyPI API token (starts with `pypi-...`)

## Verify It Worked

```powershell
# Test install in fresh environment
python -m venv test_install
test_install\Scripts\Activate.ps1
pip install autotsforecast[all]
python -c "import autotsforecast; print(f'Success! Version: {autotsforecast.__version__}')"
deactivate
Remove-Item -Recurse -Force test_install
```

## That's It!

Users can now run:
```bash
pip install autotsforecast[all]
```

---

## Testing First (Recommended)

If you want to test on TestPyPI first:

```powershell
# Upload to test server
python -m twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ autotsforecast[all]
```

TestPyPI credentials at: https://test.pypi.org/account/register/
