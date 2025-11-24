# Automated CI/CD with Semantic Versioning

This repository uses **automated semantic versioning** and **CI/CD** for seamless releases and PyPI publishing.

## 🚀 How It Works

### Workflow Overview

```
develop branch → Pull Request → main branch → Auto Version → Auto Release → Auto Publish to PyPI
```

### Branching Strategy

1. **`develop`** - Active development branch
2. **`main`** - Production-ready code with automatic releases
3. **Feature branches** - Branch from `develop`, merge back to `develop`

## 📝 Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) to automatically determine version bumps:

### Commit Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Commit Types & Version Bumps

| Commit Type | Version Bump | Example | Result |
|-------------|--------------|---------|--------|
| `fix:` | **PATCH** (0.0.X) | `fix: correct tree traversal bug` | 1.0.0 → 1.0.1 |
| `feat:` | **MINOR** (0.X.0) | `feat: add binary search tree` | 1.0.0 → 1.1.0 |
| `BREAKING CHANGE:` or `feat!:` | **MAJOR** (X.0.0) | `feat!: change API structure` | 1.0.0 → 2.0.0 |
| `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `chore:` | **PATCH** | `docs: update README` | 1.0.0 → 1.0.1 |

### Examples

```bash
# Patch version bump (1.0.0 -> 1.0.1)
git commit -m "fix: correct edge case in graph traversal"
git commit -m "docs: update installation instructions"

# Minor version bump (1.0.0 -> 1.1.0)
git commit -m "feat: add AVL tree implementation"
git commit -m "feat(trees): add tree balancing methods"

# Major version bump (1.0.0 -> 2.0.0)
git commit -m "feat!: change Graph API to use node IDs"
git commit -m "feat: redesign API

BREAKING CHANGE: Graph now requires node IDs instead of labels"
```

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Make sure you're on develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/new-tree-algorithm
```

### 2. Make Changes with Conventional Commits

```bash
# Make your changes
# ...

# Commit with conventional commit format
git add .
git commit -m "feat: add red-black tree implementation"
git commit -m "test: add tests for red-black tree"
git commit -m "docs: document red-black tree usage"
```

### 3. Push and Create Pull Request to `develop`

```bash
git push origin feature/new-tree-algorithm
```

- Create PR from `feature/new-tree-algorithm` → `develop`
- Get code review
- Merge to `develop`

### 4. Create Release (Merge to `main`)

When ready to release:

1. Create PR from `develop` → `main`
2. The PR will show what version bump will occur
3. Merge the PR
4. **Automation happens:**
   - ✅ Tests run
   - ✅ Version auto-bumps in `setup.py` and `__init__.py`
   - ✅ Git tag created (e.g., `v1.2.0`)
   - ✅ GitHub Release created with changelog
   - ✅ Package published to PyPI

## 🎯 Manual Version Control

If you need to manually specify the version bump:

1. Go to Actions tab on GitHub
2. Select "Release and Publish" workflow
3. Click "Run workflow"
4. Choose branch: `main`
5. Select version bump type: `patch`, `minor`, or `major`
6. Click "Run workflow"

## 🔍 What Happens Automatically

### On Push to `main` (or merged PR)

The `release.yml` workflow:

1. **Analyzes commits** since last release
2. **Determines version bump** based on conventional commits
3. **Runs all tests** - must pass to continue
4. **Updates version** in `setup.py` and `__init__.py`
5. **Commits version bump** back to repository
6. **Creates Git tag** (e.g., `v1.2.3`)
7. **Generates release notes** from commit messages
8. **Creates GitHub Release** with notes
9. **Builds package** (`python -m build`)
10. **Publishes to PyPI** using stored API token

### On Pull Requests

The `pr-checks.yml` workflow:

1. **Runs tests** on Python 3.9, 3.10, 3.11, 3.12
2. **Lints code** with flake8
3. **Checks commit messages** for conventional format
4. **Shows version bump preview** (what version will be released)

## 📦 Version Bump Examples

Starting version: **1.5.3**

| Commits | Bump Type | New Version |
|---------|-----------|-------------|
| `fix: bug in linked list` | PATCH | 1.5.4 |
| `feat: add hash table` | MINOR | 1.6.0 |
| `feat!: redesign tree API` | MAJOR | 2.0.0 |
| `fix: typo`<br>`feat: add queue` | MINOR | 1.6.0 |
| `feat: feature 1`<br>`feat!: breaking change` | MAJOR | 2.0.0 |

## 🛠️ Setup Requirements

### GitHub Secrets

Make sure you have set up in repository settings → Secrets:

- `PYPI_API_TOKEN` - Your PyPI API token for publishing

### Branch Protection (Recommended)

For `main` branch:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date

For `develop` branch:
- Require status checks to pass

## 📋 Checklist for Contributors

- [ ] Use conventional commit format
- [ ] Work on feature branches from `develop`
- [ ] Create PR to `develop` for features
- [ ] Create PR from `develop` to `main` for releases
- [ ] Ensure all tests pass
- [ ] Update documentation if needed

## 🚫 What NOT to Do

- ❌ Don't manually update version in `setup.py`
- ❌ Don't manually create tags
- ❌ Don't push directly to `main`
- ❌ Don't manually publish to PyPI

## 🆘 Troubleshooting

### Release wasn't created

- Check if commits follow conventional format
- Check GitHub Actions logs for errors
- Ensure PYPI_API_TOKEN secret is set
- Ensure tests pass

### Wrong version bump

- Review commit messages
- Use correct prefixes: `fix:` for patch, `feat:` for minor
- Use `BREAKING CHANGE:` or `!` for major

### Manual override needed

- Use workflow_dispatch (see "Manual Version Control" section)
- Or create a commit with the desired prefix

## 📚 Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
