import os

# run 10 trials of nn (tuned) in sequence
for i in range(10):
    cmd = f"python run.py --settings_file settings/s2.json --seed {i}"
    os.system(cmd)
