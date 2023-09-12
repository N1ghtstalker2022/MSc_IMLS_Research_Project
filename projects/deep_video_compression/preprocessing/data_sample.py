import random

# create a list with numbers from 1 to 1000
numbers = list(range(1, 1001))

# randomly sample 80% of the numbers (800 numbers) for the train set
train_numbers = random.sample(numbers, int(0.8 * len(numbers)))
train_numbers.sort()

# the rest numbers (200 numbers) are for the test set
val_numbers = [n for n in numbers if n not in train_numbers]
test_numbers = list(range(1, 11))

train_filename = "/scratch/zczqyc4/vimeo_septuplet/sep_small_trainlist.txt"
val_filename = "/scratch/zczqyc4/vimeo_septuplet/sep_small_testlist.txt"
test_filename = "/scratch/zczqyc4/vimeo_septuplet/sep_for_testlist.txt"

# write train numbers to the train file
with open(train_filename, "w") as file:
    for i in train_numbers:
        line = f"00001/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00002/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00003/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00004/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00005/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00006/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00007/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00008/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00009/{str(i).zfill(4)}\n"
        file.write(line)
    for i in train_numbers:
        line = f"00011/{str(i).zfill(4)}\n"
        file.write(line)

# write test numbers to the test file
with open(val_filename, "w") as file:
    for i in val_numbers:
        line = f"00001/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00002/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00003/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00004/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00005/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00006/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00007/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00008/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00009/{str(i).zfill(4)}\n"
        file.write(line)
    for i in val_numbers:
        line = f"00011/{str(i).zfill(4)}\n"
        file.write(line)

with open(test_filename, "w") as file:
    for i in test_numbers:
        line = f"00012/{str(i).zfill(4)}\n"
        file.write(line)