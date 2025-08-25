#!/usr/bin/env python3
"""
Efficient Git Pusher Agent
Automates the complete git workflow: add + commit + push with proper error handling.
"""

import subprocess
import sys
import argparse
from typing import Tuple, Optional


class GitPusher:
    """Efficient git automation agent that handles the complete workflow."""
    
    def __init__(self):
        self.branch_name = self._get_current_branch()
    
    def _run_command(self, command: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Execute a shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result.returncode == 0, result.stdout.strip() if capture_output else ""
        except Exception as e:
            return False, str(e)
    
    def _get_current_branch(self) -> str:
        """Get the current git branch name."""
        success, output = self._run_command("git rev-parse --abbrev-ref HEAD")
        return output if success else "main"
    
    def check_git_status(self) -> Tuple[str, str, str]:
        """Check git status, diff, and recent commits."""
        print("🔍 Checking repository status...")
        
        # Get git status
        _, status = self._run_command("git status --porcelain")
        
        # Get diff for staged and unstaged changes
        _, diff = self._run_command("git diff HEAD")
        
        # Get recent commit history for message style reference
        _, log = self._run_command("git log --oneline -5")
        
        return status, diff, log
    
    def generate_commit_message(self, status: str, diff: str) -> str:
        """Generate an appropriate commit message based on changes."""
        if not status and not diff:
            return "Minor updates and improvements"
        
        # Analyze changes to determine commit type
        lines = status.split('\n') if status else []
        added_files = [line for line in lines if line.startswith('??')]
        modified_files = [line for line in lines if line.startswith(' M') or line.startswith('M')]
        deleted_files = [line for line in lines if line.startswith(' D') or line.startswith('D')]
        
        commit_types = []
        if added_files:
            commit_types.append("add new files")
        if modified_files:
            commit_types.append("update existing files")
        if deleted_files:
            commit_types.append("remove files")
        
        if not commit_types:
            commit_types.append("update project")
        
        base_message = " and ".join(commit_types).capitalize()
        
        # Add specific details if possible
        if len(lines) <= 5 and lines:
            # For small changesets, be more specific
            file_names = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        file_names.append(parts[-1].split('/')[-1])  # Get just filename
            
            if file_names and len(commit_types) == 1:
                # Only override if there's a single type of change and few files
                if len(file_names) == 1:
                    if added_files:
                        base_message = f"Add {file_names[0]}"
                    elif modified_files:
                        base_message = f"Update {file_names[0]}"
                    elif deleted_files:
                        base_message = f"Remove {file_names[0]}"
                elif len(file_names) <= 3:
                    if added_files:
                        base_message = f"Add {', '.join(file_names)}"
                    elif modified_files:
                        base_message = f"Update {', '.join(file_names)}"
                    elif deleted_files:
                        base_message = f"Remove {', '.join(file_names)}"
        
        return base_message
    
    def execute_git_workflow(self, commit_message: str) -> bool:
        """Execute the complete git workflow: add + commit + push."""
        print("🚀 Executing git workflow...")
        
        # Create the complete commit message with required footer
        full_commit_message = f"""{commit_message}

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
        
        # Use heredoc to handle multi-line commit message properly
        heredoc_command = f"""git add . && git commit -m "$(cat <<'EOF'
{full_commit_message}
EOF
)" && git push"""
        
        print("📝 Adding files, committing, and pushing...")
        success, output = self._run_command(heredoc_command)
        
        if success:
            print("✅ Successfully completed git workflow!")
            return True
        
        # If push failed, check if it's due to no upstream branch
        if "no upstream branch" in output.lower() or "set-upstream" in output.lower():
            print("🔄 Setting upstream branch and retrying push...")
            upstream_command = f"git push --set-upstream origin {self.branch_name}"
            success, output = self._run_command(upstream_command)
            
            if success:
                print("✅ Successfully set upstream and pushed!")
                return True
            else:
                print(f"❌ Failed to push with upstream: {output}")
                return False
        
        print(f"❌ Git workflow failed: {output}")
        return False
    
    def verify_completion(self) -> None:
        """Verify the git workflow completed successfully."""
        print("🔍 Verifying completion...")
        success, status = self._run_command("git status")
        
        if success:
            print("📊 Final status:")
            print(status)
        else:
            print("⚠️  Could not verify final status")
    
    def run(self, custom_message: Optional[str] = None) -> bool:
        """Run the complete efficient git pusher workflow."""
        print("🤖 Git Pusher Agent - Efficient Git Automation")
        print("=" * 50)
        
        try:
            # Step 1: Check repository status
            status, diff, log = self.check_git_status()
            
            # Show recent commits for context
            if log:
                print("📜 Recent commits:")
                for line in log.split('\n')[:3]:  # Show last 3 commits
                    print(f"  {line}")
                print()
            
            # Step 2: Check if there are any changes
            if not status.strip() and not diff.strip():
                print("ℹ️  No changes detected. Repository is clean.")
                return True
            
            # Step 3: Generate or use custom commit message
            if custom_message:
                commit_message = custom_message
                print(f"📝 Using custom commit message: {commit_message}")
            else:
                commit_message = self.generate_commit_message(status, diff)
                print(f"📝 Generated commit message: {commit_message}")
            
            # Step 4: Execute the complete workflow
            success = self.execute_git_workflow(commit_message)
            
            if success:
                # Step 5: Verify completion
                self.verify_completion()
                return True
            else:
                print("❌ Git workflow failed!")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False


def main():
    """Main entry point for the git pusher agent."""
    parser = argparse.ArgumentParser(
        description="Efficient Git Pusher Agent - Automated add+commit+push workflow"
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Custom commit message (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Showing what would be executed:")
        pusher = GitPusher()
        status, diff, log = pusher.check_git_status()
        
        if args.message:
            commit_message = args.message
        else:
            commit_message = pusher.generate_commit_message(status, diff)
        
        print(f"Would execute: git add . && git commit -m '{commit_message}' && git push")
        return
    
    # Execute the git pusher
    pusher = GitPusher()
    success = pusher.run(args.message)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()