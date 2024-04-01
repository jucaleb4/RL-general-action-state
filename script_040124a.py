import os

# run 10 trials of linear function approx with rkhs in parallel
cmd = "python run.py --settings_file settings/s1.json --parallel"
os.system(cmd)
