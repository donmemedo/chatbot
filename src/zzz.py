import json

file1 = open("../test/logger.log", "r")
file2 = open("../test/loggger.json", "a")
Lines = file1.readlines()

count = 0
fppp = []
# json.load(file1)
# Strips the newline character
for line in Lines:
    file2.write(f"{json.loads(line)}\n")
    # file2.write(str(json.loads(line)))
#     count += 1
#
#     # print(json.loads(line))
#     fppp.append(json.loads(line))
#     json.dump(json.loads(line), file2)
# print("rrr")
file2.close()
# print("Line{}: {}".format(count, line.strip()))
