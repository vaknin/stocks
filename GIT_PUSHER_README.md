# Git Pusher Agent - Efficient Git Automation

An efficient git automation agent that handles the complete workflow: `add + commit + push` with proper error handling and smart commit message generation.

## Key Features

### ✅ Problems Fixed:
1. **No venv sourcing**: Git operations work independently of Python virtual environments
2. **Complete workflow**: Always executes the full add → commit → push sequence
3. **Single command chain**: Uses `git add . && git commit -m "..." && git push` for efficiency
4. **Upstream handling**: Automatically handles `--set-upstream origin <branch>` when needed
5. **Smart error recovery**: Retries with upstream setup if initial push fails

### 🚀 Efficient Workflow:
1. Analyzes current repository status with `git status` and `git diff`
2. Checks recent commit history for message style consistency
3. Generates intelligent commit messages based on change types
4. Executes complete workflow: `git add . && git commit -m "message" && git push`
5. Handles upstream branch setup automatically if needed
6. Verifies completion with final status check

## Usage

### Basic Usage
```bash
# Auto-generate commit message and push
python git_pusher.py

# Use custom commit message
python git_pusher.py -m "Fix authentication bug"

# Dry run to see what would be executed
python git_pusher.py --dry-run
```

### Command Line Options
```bash
python git_pusher.py [OPTIONS]

Options:
  -m, --message TEXT    Custom commit message (optional)
  --dry-run            Show what would be done without executing
  -h, --help           Show help message
```

## Smart Commit Message Generation

The agent automatically generates appropriate commit messages based on the types of changes detected:

- **New files**: `Add new_file.py` or `Add new files`
- **Modified files**: `Update config.py` or `Update existing files`
- **Deleted files**: `Remove old_file.py` or `Remove files`
- **Mixed changes**: `Add new files and update existing files`
- **No changes**: `Minor updates and improvements`

For small changesets (≤5 files), it includes specific filenames. For larger changesets, it uses generic descriptions.

## Example Workflows

### Scenario 1: Single File Update
```bash
$ python git_pusher.py
🤖 Git Pusher Agent - Efficient Git Automation
==================================================
🔍 Checking repository status...
📜 Recent commits:
  abc123 Previous commit message
  
📝 Generated commit message: Update config.py
🚀 Executing git workflow...
📝 Adding files, committing, and pushing...
✅ Successfully completed git workflow!
```

### Scenario 2: Multiple Changes with Custom Message
```bash
$ python git_pusher.py -m "Implement user authentication system"
🤖 Git Pusher Agent - Efficient Git Automation
==================================================
🔍 Checking repository status...
📝 Using custom commit message: Implement user authentication system
🚀 Executing git workflow...
✅ Successfully completed git workflow!
```

### Scenario 3: New Branch (Upstream Setup)
```bash
$ python git_pusher.py
🤖 Git Pusher Agent - Efficient Git Automation
==================================================
🔍 Checking repository status...
📝 Generated commit message: Add new feature
🚀 Executing git workflow...
🔄 Setting upstream branch and retrying push...
✅ Successfully set upstream and pushed!
```

## Error Handling

The agent includes robust error handling:

- **No changes**: Gracefully exits if repository is clean
- **Push failures**: Automatically retries with upstream setup
- **Command failures**: Provides clear error messages
- **Keyboard interrupt**: Handles Ctrl+C gracefully

## Commit Message Format

All commit messages include the standard footer:
```
Your commit message here

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Testing

Run the test suite to verify functionality:
```bash
python test_git_pusher.py
```

The test suite includes:
- Unit tests for all core functionality
- Integration tests with actual git repository
- Commit message generation validation
- Error handling verification

## Benefits Over Manual Git Commands

1. **Efficiency**: Single command vs. multiple git commands
2. **Consistency**: Standardized commit message format
3. **Reliability**: Automatic error recovery and upstream setup
4. **Intelligence**: Context-aware commit message generation
5. **Safety**: Dry run mode for verification before execution

## Requirements

- Python 3.6+
- Git repository (initialized)
- No additional dependencies required

The agent is completely self-contained and doesn't require any Python packages beyond the standard library.