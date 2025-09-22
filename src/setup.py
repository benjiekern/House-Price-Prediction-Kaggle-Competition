import subprocess

print("Installing requirements...")
subprocess.check_call(["pip", "install", "-r", "../requirements.txt"])

print("Setup complete!")