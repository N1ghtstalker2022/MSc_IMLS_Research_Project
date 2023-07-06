filename = "/scratch/zczqyc4/vimeo_septuplet/sep_small_trainlist.txt"

with open(filename, "w") as file:
    for i in range(1, 1001):
        line = f"00001/{str(i).zfill(4)}\n"
        file.write(line)
