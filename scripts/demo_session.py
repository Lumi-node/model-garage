#!/usr/bin/env python3
"""
Simulated demo session for recording.
Runs actual garage commands with typewriter effect.
"""

import sys
import time
import subprocess


def type_command(cmd, delay=0.04):
    """Simulate typing a command."""
    sys.stdout.write("$ ")
    sys.stdout.flush()
    for char in cmd:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(0.3)


def run(cmd):
    """Run a command and show output."""
    subprocess.run(cmd, shell=True)
    time.sleep(1.5)
    print()


def main():
    time.sleep(0.5)

    # Show version
    type_command("garage --version")
    run("garage --version")

    # Open a model
    type_command("garage open gpt2")
    run("garage open gpt2")

    # Extract components from layer 6
    type_command("garage extract gpt2 --layer 6")
    run("garage extract gpt2 --layer 6")

    # Analyze activations
    type_command('garage analyze gpt2 --prompt "The meaning of life"')
    run('garage analyze gpt2 --prompt "The meaning of life"')

    # Compare two models
    type_command("garage compare gpt2 distilgpt2")
    run("garage compare gpt2 distilgpt2")

    time.sleep(2)


if __name__ == "__main__":
    main()
