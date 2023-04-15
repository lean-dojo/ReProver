import sys

num_total = num_correct = 0

for line in open(sys.argv[1]):
    if "SearchResult" in line:
        num_total += 1
        if "Proved" in line:
            num_correct += 1

print(f"{num_correct} / {num_total} = {num_correct / num_total}")
