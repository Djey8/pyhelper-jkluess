#!/usr/bin/env python3
"""
Validate commit messages follow Conventional Commits format
Usage: python validate_commits.py
"""

import re
import subprocess
import sys

# Conventional commit pattern
PATTERN = re.compile(
    r'^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)'
    r'(\([a-z\-]+\))?'
    r'(!)?'
    r': .+'
)

def get_commits_since_last_tag():
    """Get commit messages since last tag"""
    try:
        # Get last tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            last_tag = result.stdout.strip()
            # Get commits since last tag
            result = subprocess.run(
                ['git', 'log', f'{last_tag}..HEAD', '--pretty=format:%s'],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # No tags, get all commits
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%s'],
                capture_output=True,
                text=True,
                check=True
            )
        
        return result.stdout.strip().split('\n') if result.stdout else []
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits: {e}")
        return []

def validate_commit(message):
    """Validate a single commit message"""
    # Check for breaking change
    if 'BREAKING CHANGE:' in message:
        return True, 'MAJOR', 'Breaking change detected'
    
    # Check for conventional commit format
    match = PATTERN.match(message)
    if match:
        commit_type = match.group(1)
        has_breaking = match.group(3) == '!'
        
        if has_breaking:
            return True, 'MAJOR', f'{commit_type}! (breaking change)'
        elif commit_type == 'feat':
            return True, 'MINOR', 'New feature'
        else:
            return True, 'PATCH', f'{commit_type.capitalize()} change'
    
    return False, None, 'Not a conventional commit'

def main():
    print("🔍 Validating Conventional Commits\n")
    print("=" * 70)
    
    commits = get_commits_since_last_tag()
    
    if not commits or (len(commits) == 1 and not commits[0]):
        print("ℹ️  No commits to validate")
        return 0
    
    valid_commits = []
    invalid_commits = []
    highest_bump = 'NONE'
    
    # Priority: MAJOR > MINOR > PATCH
    bump_priority = {'MAJOR': 3, 'MINOR': 2, 'PATCH': 1, 'NONE': 0}
    
    for commit in commits:
        is_valid, bump_type, reason = validate_commit(commit)
        
        if is_valid:
            valid_commits.append((commit, bump_type, reason))
            if bump_priority.get(bump_type, 0) > bump_priority[highest_bump]:
                highest_bump = bump_type
        else:
            invalid_commits.append((commit, reason))
    
    # Print valid commits
    if valid_commits:
        print(f"\n✅ Valid Conventional Commits ({len(valid_commits)}):\n")
        for commit, bump_type, reason in valid_commits:
            icon = {
                'MAJOR': '🔴',
                'MINOR': '🟢',
                'PATCH': '🔵'
            }.get(bump_type, '⚪')
            print(f"  {icon} [{bump_type:5}] {commit}")
            print(f"           → {reason}")
    
    # Print invalid commits
    if invalid_commits:
        print(f"\n⚠️  Non-conventional Commits ({len(invalid_commits)}):\n")
        for commit, reason in invalid_commits:
            print(f"  ❌ {commit}")
            print(f"     → {reason}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("\n📊 Summary:\n")
    print(f"  Total commits: {len(commits)}")
    print(f"  Valid:   {len(valid_commits)}")
    print(f"  Invalid: {len(invalid_commits)}")
    
    if highest_bump != 'NONE':
        icon = {
            'MAJOR': '🔴',
            'MINOR': '🟢',
            'PATCH': '🔵'
        }[highest_bump]
        print(f"\n{icon} Version Bump: {highest_bump}")
        
        if highest_bump == 'MAJOR':
            print("  ├─ Breaking changes detected!")
            print("  └─ Will bump: X.0.0")
        elif highest_bump == 'MINOR':
            print("  ├─ New features detected")
            print("  └─ Will bump: 0.X.0")
        else:
            print("  ├─ Fixes/improvements detected")
            print("  └─ Will bump: 0.0.X")
    else:
        print("\n⚠️  No conventional commits found")
        print("  └─ No release will be created")
    
    # Print recommendations
    if invalid_commits:
        print("\n💡 Recommendations:\n")
        print("  Use conventional commit format:")
        print("    feat:     New feature (minor bump)")
        print("    fix:      Bug fix (patch bump)")
        print("    docs:     Documentation (patch bump)")
        print("    test:     Tests (patch bump)")
        print("    refactor: Code refactoring (patch bump)")
        print("    feat!:    Breaking change (major bump)")
        print("\n  Example: feat(trees): add AVL tree implementation")
    
    print("\n" + "=" * 70)
    
    # Return exit code
    if not commits or (len(commits) == 1 and not commits[0]):
        return 0
    elif invalid_commits and not valid_commits:
        print("\n❌ No valid conventional commits found")
        return 1
    else:
        print("\n✅ Validation complete")
        return 0

if __name__ == '__main__':
    sys.exit(main())
