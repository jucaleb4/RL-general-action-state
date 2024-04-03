import os
# https://stackoverflow.com/questions/21735859/convert-a-string-which-is-a-list-into-a-proper-list-python
import ast
import numpy as np
import pandas as pd
import scipy as sp

"""
Reads in logs/policy_progress.csv (assumed that PMD creates this) and parses
the changes in Q's, log(pi), and pi. We then find the 50 iteration-state pairs
that produced the biggest change in policy (wrt KL divergence)
"""

def kl(x,y):
    return np.sum(np.multiply(y, np.log(np.divide(y, x.astype("float")))))

fname = os.path.join("logs", "policy_progress.csv")
df = pd.read_csv(fname, header="infer", sep=";")
n = len(df["iter"])
# since -1 index based
max_iter = int(np.max(df["iter"]))+2
assert n % max_iter == 0, f"Got n={n} and max_iter={max_iter}"
num_s = int(n/max_iter)

s = np.array(df["s"])
q_s = np.array(df["q_s"])
logpi_s = np.array(df["logpi_s"])
pi_s = np.array(df["pi_s"])
kl_divs = np.zeros(n)

for i in range(num_s):
    print("-"*30)
    print(f"State: {s[i]}")
    assert (s[i::num_s] == s[i]).all()
    print(f"q_sa   : {q_s[i::num_s]}")
    print(f"logpi_s: {logpi_s[i::num_s]}")
    print(f"pi_s   : {pi_s[i::num_s]}")
    print("-"*30)

    for j in range(i,len(q_s)-num_s,num_s):
        pi = np.array(ast.literal_eval(pi_s[j]))
        pi_next = np.array(ast.literal_eval(pi_s[j+num_s]))
        kl_divs[j] = kl(pi, pi_next)

print("="*30)
print("="*30)
# find points where kl divergence changed the most find top 50
sorted_idxs = np.argsort(kl_divs)[-50:][::-1]
for rk, i in enumerate(sorted_idxs):
    print(f"rk {rk+1} (iter {int(i/num_s)}) State: {s[i]}")
    print(f"q_sa   : {q_s[i]} -> {q_s[i+num_s]}")
    print(f"logpi_s: {logpi_s[i]} -> {logpi_s[i+num_s]}")
    print(f"pi_s   : {pi_s[i]} -> {pi_s[i+num_s]}")
