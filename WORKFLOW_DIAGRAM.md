# CI/CD Workflow Diagram

## Complete Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   develop    │  ◄─── Feature branches merge here
└──────┬───────┘
       │
       │ ┌─────────────────────────────────────────────────────────┐
       │ │ feature/add-tree                                         │
       │ │   │                                                      │
       │ │   ├─ feat(trees): add binary search tree                │
       │ │   ├─ test(trees): add BST tests                         │
       │ │   └─ docs(trees): document BST                          │
       │ │                                                          │
       │ │   Pull Request → develop                                │
       │ │   ✓ Tests run (Python 3.9, 3.10, 3.11, 3.12)          │
       │ │   ✓ Linting passes                                     │
       │ └────────────────────────────────────────┬────────────────┘
       │                                           │
       ◄───────────────────────────────────────────┘ Merge
       │
       │ Multiple features accumulate on develop...
       │
       │ Ready for release? Create PR: develop → main
       ▼
┌──────────────┐
│     main     │  ◄─── Releases happen here
└──────┬───────┘
       │
       │ ┌─────────────────────────────────────────────────────────┐
       │ │ Pull Request: develop → main                            │
       │ │                                                          │
       │ │   PR Checks:                                            │
       │ │   ✓ Commit format validation                           │
       │ │   ✓ Version bump preview shown                         │
       │ │     → "This will bump to v1.5.0 (MINOR)"               │
       │ │   ✓ All tests pass                                     │
       │ └────────────────────────────────────┬────────────────────┘
       │                                       │
       ◄───────────────────────────────────────┘ Merge triggers automation!
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTOMATED RELEASE WORKFLOW                        │
│                                                                      │
│  1. 🧪 Run Tests                                                    │
│     └─ pytest tests/ -v (must pass to continue)                    │
│                                                                      │
│  2. 📊 Analyze Commits                                             │
│     ├─ Get commits since last tag                                  │
│     ├─ Detect: feat → MINOR (0.X.0)                               │
│     ├─ Detect: fix → PATCH (0.0.X)                                │
│     └─ Detect: BREAKING CHANGE → MAJOR (X.0.0)                    │
│                                                                      │
│  3. 📝 Update Version                                              │
│     ├─ Update setup.py: version="1.5.0"                           │
│     ├─ Update __init__.py: __version__ = "1.5.0"                  │
│     └─ Commit: "chore: bump version to 1.5.0"                     │
│                                                                      │
│  4. 🏷️  Create Git Tag                                             │
│     └─ git tag -a v1.5.0 -m "Release v1.5.0"                      │
│                                                                      │
│  5. 📋 Generate Release Notes                                      │
│     ├─ Extract commits since last version                          │
│     ├─ Format as changelog                                         │
│     └─ Include comparison link                                     │
│                                                                      │
│  6. 🎉 Create GitHub Release                                       │
│     ├─ Tag: v1.5.0                                                │
│     ├─ Title: "Release v1.5.0"                                    │
│     └─ Body: Generated release notes                              │
│                                                                      │
│  7. 📦 Build Package                                               │
│     └─ python -m build (creates dist/*.whl and dist/*.tar.gz)     │
│                                                                      │
│  8. 🚀 Publish to PyPI                                             │
│     └─ twine upload dist/*                                         │
│         (uses PYPI_API_TOKEN secret)                               │
│                                                                      │
│  ✅ DONE! Package version 1.5.0 is now live on PyPI               │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
   Users can now:
   pip install pyhelper-jkluess==1.5.0


═══════════════════════════════════════════════════════════════════════════

COMMIT MESSAGE → VERSION BUMP EXAMPLES

Example 1: Patch Release (Bug Fix)
┌────────────────────────────────────────────────────────┐
│ Commits:                                                │
│   fix(graphs): correct cycle detection bug             │
│   test(graphs): add cycle detection test               │
│                                                         │
│ Version: 1.2.3 → 1.2.4 (PATCH)                        │
└────────────────────────────────────────────────────────┘

Example 2: Minor Release (New Feature)
┌────────────────────────────────────────────────────────┐
│ Commits:                                                │
│   feat(trees): add AVL tree implementation             │
│   test(trees): add AVL tree tests                      │
│   docs(trees): document AVL tree usage                 │
│                                                         │
│ Version: 1.2.4 → 1.3.0 (MINOR)                        │
└────────────────────────────────────────────────────────┘

Example 3: Major Release (Breaking Change)
┌────────────────────────────────────────────────────────┐
│ Commits:                                                │
│   feat!: redesign Graph API for performance            │
│                                                         │
│   BREAKING CHANGE: Graph constructor now requires      │
│   graph_type parameter                                 │
│                                                         │
│ Version: 1.3.0 → 2.0.0 (MAJOR)                        │
└────────────────────────────────────────────────────────┘

Example 4: Mixed Commits (Highest Priority Wins)
┌────────────────────────────────────────────────────────┐
│ Commits:                                                │
│   docs: update README                    (patch)       │
│   fix(lists): correct memory leak        (patch)       │
│   feat(skiplist): add iterator           (minor) ◄─┐   │
│   test: add more tests                   (patch)    │   │
│                                                      │   │
│ Version: 1.3.0 → 1.4.0 (MINOR)                     │   │
│ Reason: feat (minor) overrides fix/docs (patch) ───┘   │
└────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════

BRANCH STRATEGY

develop (active development)
  │
  ├─ feature/add-hash-table
  │    └─ feat(structures): add hash table
  │         │
  │         └─ PR to develop → Merge
  │
  ├─ fix/graph-bug
  │    └─ fix(graphs): correct Dijkstra
  │         │
  │         └─ PR to develop → Merge
  │
  └─ feature/improve-docs
       └─ docs: improve documentation
            │
            └─ PR to develop → Merge

When ready to release:
  PR: develop → main → Triggers automatic release


═══════════════════════════════════════════════════════════════════════════

MANUAL OVERRIDE (if needed)

GitHub Actions → "Release and Publish" → "Run workflow"

┌──────────────────────────────────────────┐
│  Run workflow                             │
│                                           │
│  Branch: main                 ▼           │
│                                           │
│  Version bump:                            │
│    ○ Auto-detect (default)               │
│    ○ patch                                │
│    ○ minor                                │
│    ○ major                                │
│                                           │
│          [ Run workflow ]                 │
└──────────────────────────────────────────┘

This bypasses commit message analysis and forces the chosen version bump.
```

## Visual Commit Type Decision Tree

```
Your commit message
        │
        ▼
Does it contain "BREAKING CHANGE:" or "feat!:" ?
        │
    ┌───┴───┐
   Yes      No
    │        │
    ▼        ▼
  MAJOR    Does it start with "feat:" ?
 (X.0.0)    │
        ┌───┴───┐
       Yes      No
        │        │
        ▼        ▼
      MINOR    Does it start with "fix:", "docs:", etc?
     (0.X.0)    │
            ┌───┴───┐
           Yes      No
            │        │
            ▼        ▼
          PATCH    No release
         (0.0.X)   (skip)
```

## Timeline Example

```
Monday
  09:00 - Developer creates feature/add-avl-tree from develop
  10:30 - Commits: feat(trees): add AVL tree
  11:00 - Commits: test(trees): add AVL tests
  14:00 - Push and create PR to develop
  14:30 - PR merged to develop

Tuesday  
  09:00 - Another developer creates fix/graph-bug from develop
  10:00 - Commits: fix(graphs): correct cycle detection
  11:00 - PR merged to develop

Wednesday
  09:00 - Team decides to release
  09:15 - Create PR: develop → main
  09:20 - Review PR (shows: "Will bump to v1.4.0 - MINOR")
  09:30 - Merge PR
  09:31 - 🤖 CI starts automatically
  09:32 - ✅ Tests pass
  09:33 - 📝 Version updated to 1.4.0
  09:34 - 🏷️  Tag v1.4.0 created
  09:35 - 📋 Release notes generated
  09:36 - 🎉 GitHub release created
  09:37 - 📦 Package built
  09:38 - 🚀 Published to PyPI
  09:39 - ✅ Done! Users can: pip install pyhelper-jkluess==1.4.0
```
