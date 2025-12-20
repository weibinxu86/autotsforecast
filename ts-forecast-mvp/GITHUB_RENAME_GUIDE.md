# GitHub Repository Rename Guide

## Renaming Your Repository to `autotsforecast`

Follow these steps to rename your GitHub repository from `ts-forecast-mvp` to `autotsforecast`:

### Option 1: Via GitHub Website (Easiest)

1. **Go to your repository** on GitHub: `https://github.com/weibinxu86/ts-forecast-mvp`

2. **Click on "Settings"** (top right of the repository page)

3. **Scroll down to "Repository name"** section

4. **Change the name** from `ts-forecast-mvp` to `autotsforecast`

5. **Click "Rename"** button

6. **GitHub will automatically redirect** all old URLs to the new name

### Option 2: Update Local Git Remote (After renaming on GitHub)

If you've already renamed on GitHub, update your local repository:

```bash
cd c:\forecasting\ts-forecast-mvp

# Update the remote URL
git remote set-url origin https://github.com/weibinxu86/autotsforecast.git

# Verify the change
git remote -v
```

Should show:
```
origin  https://github.com/weibinxu86/autotsforecast.git (fetch)
origin  https://github.com/weibinxu86/autotsforecast.git (push)
```

### Option 3: Rename Local Folder (Optional)

If you also want to rename your local project folder:

```powershell
# Navigate to parent directory
cd c:\forecasting

# Rename the folder
Rename-Item -Path "ts-forecast-mvp" -NewName "autotsforecast"

# Navigate to new folder
cd autotsforecast
```

### Important Notes

✅ **GitHub handles redirects**: Old URLs automatically redirect to new name
✅ **Clones still work**: Existing clones will continue to work
✅ **CI/CD**: Update any CI/CD pipelines or deployment scripts
✅ **Documentation**: All docs in this repo already updated to use `weibinxu86/autotsforecast`

### What's Been Updated

All references in your codebase have been updated:

- ✅ `setup.py` - Author: Weibin Xu, URL: weibinxu86/autotsforecast
- ✅ `pyproject.toml` - Repository: weibinxu86/autotsforecast
- ✅ `README.md` - Clone URL, Issues link, Citation
- ✅ All other documentation files

### After Renaming

1. **Test the installation**:
   ```bash
   pip install git+https://github.com/weibinxu86/autotsforecast.git
   ```

2. **Update any external references**:
   - Documentation sites
   - Blog posts
   - Papers citing the code

3. **Notify users** (if public):
   - Create a release note mentioning the rename
   - Update PyPI description when publishing

### Verification Checklist

- [ ] Repository renamed on GitHub
- [ ] Local git remote updated
- [ ] Can clone from new URL
- [ ] All links in README work
- [ ] CI/CD pipelines updated (if any)
- [ ] PyPI metadata will use new URL

## Your New Repository URLs

- **Repository**: https://github.com/weibinxu86/autotsforecast
- **Clone (HTTPS)**: `git clone https://github.com/weibinxu86/autotsforecast.git`
- **Clone (SSH)**: `git clone git@github.com:weibinxu86/autotsforecast.git`
- **Issues**: https://github.com/weibinxu86/autotsforecast/issues
- **Discussions**: https://github.com/weibinxu86/autotsforecast/discussions

---

**Ready to rename!** Just go to Settings → Repository name → `autotsforecast` → Rename
