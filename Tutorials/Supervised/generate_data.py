import numpy as np

with open("ising1d_train_samples.txt", "w") as f:
    for i in range(1024):
        d = format(i, '010b')

        outstr = ""
        for i in range(9):
            outstr += d[i] + " "
        outstr += d[-1] + "\n"
        f.write(outstr)
