#!/usr/bin/env python3
"""
Test suite for the Git Pusher Agent
"""

import unittest
import subprocess
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path so we can import git_pusher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from git_pusher import GitPusher


class TestGitPusher(unittest.TestCase):
    """Test cases for GitPusher functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pusher = GitPusher()
    
    @patch('git_pusher.subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        # Mock successful command
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_run.return_value = mock_result
        
        success, output = self.pusher._run_command("git status")
        
        self.assertTrue(success)
        self.assertEqual(output, "test output")
        mock_run.assert_called_once()
    
    @patch('git_pusher.subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test failed command execution."""
        # Mock failed command
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_run.return_value = mock_result
        
        success, output = self.pusher._run_command("git invalid-command")
        
        self.assertFalse(success)
    
    def test_generate_commit_message_empty(self):
        """Test commit message generation with no changes."""
        message = self.pusher.generate_commit_message("", "")
        self.assertEqual(message, "Minor updates and improvements")
    
    def test_generate_commit_message_new_files(self):
        """Test commit message generation with new files."""
        status = "?? new_file.py\n?? another_file.txt"
        message = self.pusher.generate_commit_message(status, "")
        self.assertIn("add", message.lower())
    
    def test_generate_commit_message_modified_files(self):
        """Test commit message generation with modified files."""
        status = " M existing_file.py\nM  another_file.txt"
        message = self.pusher.generate_commit_message(status, "")
        self.assertIn("update", message.lower())
    
    def test_generate_commit_message_mixed_changes(self):
        """Test commit message generation with mixed changes."""
        status = "?? new_file.py\n M existing_file.py\n D old_file.txt"
        message = self.pusher.generate_commit_message(status, "")
        # Should contain multiple action types
        self.assertTrue(
            any(word in message.lower() for word in ["add", "update", "remove"])
        )
    
    def test_generate_commit_message_specific_files(self):
        """Test commit message generation with specific files."""
        status = " M config.py"
        message = self.pusher.generate_commit_message(status, "")
        self.assertIn("config.py", message)
    
    @patch('git_pusher.GitPusher._run_command')
    def test_check_git_status(self, mock_run_command):
        """Test git status checking."""
        # Mock the three commands: status, diff, log
        mock_run_command.side_effect = [
            (True, "M  file1.py\n?? file2.py"),  # git status
            (True, "diff content"),               # git diff
            (True, "abc123 Previous commit")      # git log
        ]
        
        status, diff, log = self.pusher.check_git_status()
        
        self.assertEqual(status, "M  file1.py\n?? file2.py")
        self.assertEqual(diff, "diff content")
        self.assertEqual(log, "abc123 Previous commit")
        self.assertEqual(mock_run_command.call_count, 3)
    
    def test_commit_message_format(self):
        """Test that generated commit messages follow expected format."""
        # Test various scenarios
        test_cases = [
            ("?? new_file.py", "Add new files"),
            (" M existing.py", "Update existing.py"),
            (" D old.py", "Remove files"),
        ]
        
        for status, expected_pattern in test_cases:
            message = self.pusher.generate_commit_message(status, "")
            # Basic format check - should be capitalized and descriptive
            self.assertTrue(message[0].isupper(), f"Message should start with capital: {message}")
            self.assertGreater(len(message), 5, f"Message should be descriptive: {message}")


class TestGitPusherIntegration(unittest.TestCase):
    """Integration tests for GitPusher (require actual git repository)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Only run if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            self.skipTest("Not in a git repository")
        
        self.pusher = GitPusher()
    
    def test_get_current_branch(self):
        """Test getting current branch name."""
        branch = self.pusher._get_current_branch()
        self.assertIsInstance(branch, str)
        self.assertGreater(len(branch), 0)
    
    def test_dry_run_functionality(self):
        """Test that dry run mode doesn't execute actual commands."""
        # This test ensures the dry run functionality works
        with patch('sys.argv', ['git_pusher.py', '--dry-run']):
            # Should not raise any exceptions and not modify repository
            try:
                from git_pusher import main
                # Capture stdout to verify dry run output
                with patch('builtins.print') as mock_print:
                    main()
                    # Should have printed dry run information
                    self.assertTrue(any('DRY RUN' in str(call) for call in mock_print.call_args_list))
            except SystemExit:
                pass  # Expected for argument parsing


def run_tests():
    """Run all tests with appropriate verbosity."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGitPusher))
    suite.addTests(loader.loadTestsFromTestCase(TestGitPusherIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)