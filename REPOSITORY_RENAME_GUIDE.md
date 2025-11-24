# Repository Rename Guide

## What Was Changed

✅ **Package structure reorganized:**
- Moved `Basic/` and `Complex/` into new `pyhelper_jkluess/` directory
- Now imports as: `from pyhelper_jkluess.Complex.Trees.tree import Tree`
- All 564 tests updated and passing ✅

✅ **Files updated:**
- All test files now use `pyhelper_jkluess` imports
- All README examples updated with correct imports
- `setup.py` URL updated to point to new repository name

## Step 1: Commit Current Changes

```bash
git add .
git commit -m "Restructure package: move modules into pyhelper_jkluess directory"
```

## Step 2: Rename Repository on GitHub

1. Go to your repository on GitHub: https://github.com/Djey8/PyHelper
2. Click on **Settings** (top right)
3. Scroll down to **Repository name**
4. Change `PyHelper` to `pyhelper-jkluess`
5. Click **Rename**

⚠️ **Note:** GitHub will automatically redirect from old URL to new URL, so existing clones will continue to work.

## Step 3: Update Local Git Remote (Optional but Recommended)

After renaming on GitHub, update your local repository:

```bash
git remote set-url origin https://github.com/Djey8/pyhelper-jkluess.git
```

Verify the change:
```bash
git remote -v
```

## Step 4: Push Changes

```bash
git push origin main
```

## Step 5: Verify Installation

After pushing, test the installation:

```bash
# Uninstall old version if installed
pip uninstall pyhelper-jkluess -y

# Install from GitHub with new name
pip install git+https://github.com/Djey8/pyhelper-jkluess.git

# Test import
python -c "from pyhelper_jkluess.Complex.Trees.tree import Tree; print('Success!')"
```

## What Users Need to Know

After this change, users should:

### Installation (PyPI)
```bash
pip install pyhelper-jkluess
```

### Import Pattern
```python
# Old way (won't work anymore)
from Complex.Trees.tree import Tree  ❌

# New way (correct)
from pyhelper_jkluess.Complex.Trees.tree import Tree  ✅

# Or use the package-level import
import pyhelper_jkluess
tree = pyhelper_jkluess.Complex.Trees.Tree("Root")  ✅
```

## Migration for Existing Users

If you have existing code using the old import pattern:

**Find and replace in your project:**
- Find: `from Basic.`
- Replace with: `from pyhelper_jkluess.Basic.`

- Find: `from Complex.`
- Replace with: `from pyhelper_jkluess.Complex.`

## Benefits of This Change

✅ **Clear package identity:** Package name matches PyPI name
✅ **No import confusion:** `import pyhelper_jkluess` clearly identifies the package
✅ **Standard Python conventions:** Package name follows PEP 8 (lowercase with underscores)
✅ **Namespace clarity:** Prevents conflicts with other packages named "Basic" or "Complex"
