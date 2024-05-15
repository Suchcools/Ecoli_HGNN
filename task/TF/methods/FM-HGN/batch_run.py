import subprocess
for i in range(101,150):
    subprocess.run(["python", "run.py", "--dataset", f"INS/{i}"])