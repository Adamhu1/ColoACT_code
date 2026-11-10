#!/usr/bin/env python3
"""
Data collection launcher that uses the correct conda environment
"""
import subprocess
import sys
import os

def main():
    # Use the colon_robot conda environment's Python interpreter
    conda_python = "/path/to/conda/env/bin/python3"

    # Path to the data_collector.py script
    script_path = "/path/to/ros2_ws/src/coloact_robot/coloact_robot/data_collector.py"

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "/path/to/ros2_ws/install/coloact_robot/lib/python3.8/site-packages:" + env.get("PYTHONPATH", "")

    # Run the data collector with the correct Python interpreter
    cmd = [conda_python, script_path] + sys.argv[1:]

    print(f"Starting data collector with conda environment...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running data collector: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nData collector stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()


