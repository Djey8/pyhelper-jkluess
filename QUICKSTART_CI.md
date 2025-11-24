# Quick Start Guide for CI/CD

## TL;DR

1. **Use conventional commits**: `feat:`, `fix:`, `docs:`, etc.
2. **Develop on `develop` branch**
3. **Merge to `main`** → Automatic release + PyPI publish

## Quick Examples

### Add a Feature (Minor Version Bump)

```bash
git checkout develop
git pull origin develop
git checkout -b feature/add-heap

# Make your changes
git add .
git commit -m "feat(trees): add min-heap and max-heap implementations"
git commit -m "test(trees): add comprehensive heap tests"
git commit -m "docs(trees): document heap usage and complexity"

git push origin feature/add-heap
# Create PR to develop → merge
```

Result when merged to main: `1.0.0` → `1.1.0`

### Fix a Bug (Patch Version Bump)

```bash
git checkout develop
git pull origin develop
git checkout -b fix/graph-cycle-detection

# Fix the bug
git add .
git commit -m "fix(graphs): correct cycle detection for disconnected graphs"
git commit -m "test(graphs): add test for cycle detection edge case"

git push origin fix/graph-cycle-detection
# Create PR to develop → merge
```

Result when merged to main: `1.0.0` → `1.0.1`

### Breaking Change (Major Version Bump)

```bash
git checkout develop
git pull origin develop
git checkout -b feature/api-redesign

# Make breaking changes
git add .
git commit -m "feat!: redesign Graph API for better performance

BREAKING CHANGE: Graph constructor now requires type parameter.
Old: Graph()
New: Graph(graph_type='undirected')"

git push origin feature/api-redesign
# Create PR to develop → merge
```

Result when merged to main: `1.5.3` → `2.0.0`

## Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Version Bump | Use Case |
|------|--------------|----------|
| `feat` | MINOR (0.X.0) | New feature |
| `fix` | PATCH (0.0.X) | Bug fix |
| `docs` | PATCH (0.0.X) | Documentation only |
| `style` | PATCH (0.0.X) | Code style (formatting) |
| `refactor` | PATCH (0.0.X) | Code refactoring |
| `perf` | PATCH (0.0.X) | Performance improvement |
| `test` | PATCH (0.0.X) | Adding tests |
| `chore` | PATCH (0.0.X) | Maintenance tasks |
| `feat!` or `BREAKING CHANGE:` | MAJOR (X.0.0) | Breaking changes |

### Scopes (Optional)

- `trees` - Tree-related changes
- `graphs` - Graph-related changes  
- `lists` - List-related changes
- `skiplist` - Skip list changes
- `ci` - CI/CD changes

## Setup Your Git for Conventional Commits

```bash
# Set commit message template
git config commit.template .gitmessage

# Now when you commit, you'll see the template
git commit
```

## Release Workflow

### Standard Release

1. Develop features on `develop` branch
2. Create PR from `develop` → `main`
3. Review and merge
4. **Automatic:**
   - Version bumped
   - Tag created
   - Release published
   - PyPI updated

### Emergency Hotfix

1. Branch from `main`: `git checkout -b hotfix/critical-bug main`
2. Fix and commit: `git commit -m "fix: critical security vulnerability"`
3. Create PR to `main`
4. Merge → auto-release

## Manual Release (If Needed)

GitHub Actions → "Release and Publish" → "Run workflow"
- Choose `main` branch
- Select version bump: patch/minor/major
- Run

## Check Before Pushing

```bash
# Check your commit messages
git log --oneline origin/develop..HEAD

# Should see:
# feat(trees): add new feature
# fix(graphs): correct bug
# NOT:
# added stuff
# fixed thing
```

## Common Scenarios

### Multiple Commits, What Version?

The **highest priority** change determines the version:

```bash
git log
# fix: bug fix          (patch)
# feat: new feature     (minor)  ← This wins
# docs: update README   (patch)
```
Result: MINOR bump

```bash
git log  
# feat: feature 1       (minor)
# feat!: breaking       (major) ← This wins
# fix: bug fix          (patch)
```
Result: MAJOR bump

### No Conventional Commits?

No release will be created. Add at least one:
```bash
git commit --allow-empty -m "chore: trigger release"
```

## Troubleshooting

### "My release wasn't created"

- Check commit messages follow conventional format
- Check GitHub Actions logs
- Ensure all tests pass

### "Wrong version number"

- Review your commit messages
- Use correct type prefix
- Remember: `feat` = minor, `fix` = patch, `feat!` = major

### "I need to skip CI"

Add `[skip ci]` to commit message:
```bash
git commit -m "docs: update README [skip ci]"
```

## Best Practices

✅ **DO:**
- Use conventional commit format
- Work on feature branches
- Test before pushing
- Write clear commit messages
- One logical change per commit

❌ **DON'T:**
- Commit directly to `main`
- Use vague messages like "fix stuff"
- Mix multiple unrelated changes
- Manually edit version numbers
- Skip tests

## Quick Reference Card

```bash
# Feature:     feat:  → 1.0.0 → 1.1.0
# Bug fix:     fix:   → 1.0.0 → 1.0.1
# Breaking:    feat!: → 1.0.0 → 2.0.0
# Docs/other:  docs:  → 1.0.0 → 1.0.1

# Workflow:
develop → PR → main → auto-release

# Check version that will be released:
# Look at PR checks - they show the bump type
```
