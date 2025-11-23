# PyPI Publishing Setup Guide

## Prerequisites

### 1. Create a PyPI Account
1. Go to [https://pypi.org/](https://pypi.org/)
2. Click "Register" to create an account
3. Verify your email address
4. Enable Two-Factor Authentication (2FA) - **required for publishing**

### 2. Create an API Token
1. Log in to PyPI
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name (e.g., "PyHelper GitHub Actions")
5. Select scope: "Entire account" (or specific project after first upload)
6. Copy the token (starts with `pypi-...`) - **you won't see it again!**

### 3. Add Token to GitHub Secrets
1. Go to your GitHub repository: https://github.com/Djey8/PyHelper
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI token (the one starting with `pypi-...`)
6. Click **Add secret**

## How to Publish Your Package

### Option 1: Automatic Publishing (via GitHub Release)
This is the **recommended** approach:

1. Make sure all your changes are committed and pushed to `main`
2. Go to your GitHub repository
3. Click **Releases** → **Create a new release**
4. Click **Choose a tag** → Type a version (e.g., `v0.1.0`) → **Create new tag**
5. Fill in the release title: "Release 0.1.0"
6. Add release notes describing what's new
7. Click **Publish release**

✨ **The GitHub Action will automatically:**
- Build your package
- Upload it to PyPI
- Make it available via `pip install pyhelper`

### Option 2: Manual Trigger
1. Go to **Actions** tab in your repository
2. Click "Publish to PyPI" workflow
3. Click **Run workflow** → **Run workflow**

### Option 3: Manual Publishing (from your computer)
```powershell
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI (you'll be prompted for credentials)
python -m twine upload dist/*
```

## Before First Upload

### Update Version Number
Edit `setup.py` and change version when releasing:
```python
version="0.1.0",  # Increment for each release: 0.1.0 → 0.1.1 → 0.2.0
```

### Test with TestPyPI First (Optional)
To test without affecting the real PyPI:

1. Create account at [https://test.pypi.org/](https://test.pypi.org/)
2. Create API token there
3. Upload test version:
```powershell
python -m twine upload --repository testpypi dist/*
```
4. Test install:
```powershell
pip install --index-url https://test.pypi.org/simple/ pyhelper
```

## After Publishing

Once published, anyone can install your package:
```bash
pip install pyhelper
```

Then use it:
```python
from Basic.linked_list import LinkedList
from Basic.double_linked_list import DoubleLinkedList
from Basic.circular_linked_list import CircularLinkedList

# Use your classes!
my_list = LinkedList([1, 2, 3])
```

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **0.1.0** → **0.1.1**: Bug fixes
- **0.1.0** → **0.2.0**: New features (backwards compatible)
- **0.1.0** → **1.0.0**: Major changes (may break compatibility)

## Troubleshooting

**"Package already exists"**: Increment version in `setup.py`
**"Invalid token"**: Check GitHub secret is named `PYPI_API_TOKEN` exactly
**"2FA required"**: Enable 2FA on your PyPI account
**"Filename already used"**: Delete `dist/` folder and rebuild

## Quick Checklist

- [ ] PyPI account created and email verified
- [ ] 2FA enabled on PyPI account
- [ ] API token created on PyPI
- [ ] API token added to GitHub secrets as `PYPI_API_TOKEN`
- [ ] Version number updated in `setup.py`
- [ ] All changes committed and pushed
- [ ] GitHub release created

That's it! Your package will be live on PyPI! 🎉
