import re
import sys
import numpy as np
from glob import glob
from loguru import logger
import matplotlib.pyplot as plt

total_time = []
TOTAL_TIME_REGEX = re.compile(r"total_time=(?P<time>.+?),")

for filename in glob(sys.argv[1]):
    logger.info(filename)
    num_total = num_correct = 0
    for line in open(filename):
        if "SearchResult" in line:
            num_total += 1
            if "Proved" in line:
                num_correct += 1
                total_time.append(float(TOTAL_TIME_REGEX.search(line)["time"]))

    if num_total == 0:
        logger.info("Pass@1: N/A")
    else:
        logger.info(f"Pass@1: {num_correct} / {num_total} = {num_correct / num_total}")

logger.info(f"Average time: {np.mean(total_time)}")

total_time.sort()
x = []
y = []
for i, t in enumerate(total_time):
    x.append(t)
    y.append(i + 1)
plt.scatter(x, y)
plt.savefig("stats.pdf")
logger.info("Figure saved to stats.pdf")
