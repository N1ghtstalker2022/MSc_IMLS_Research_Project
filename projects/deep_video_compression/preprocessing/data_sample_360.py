import random

classes = ["AcerPredator", "BlueWorld", "Cryogenian", "Egypt", "Graffiti",
           "AirShow", "CMLauncher", "Dota2", "F5Fighter", "Help",
           "BFG", "CS", "DrivingInAlps", "GalaxyOnFire", "HondaF1", "BTSRun",
           "CandyCarnival", "Dubai", "Gliding", "IRobot", "KasabianLive", "LOL"]
# # create a list with numbers from 1 to 1000
# numbers = list(range(1, 1001))
#
# # randomly sample 80% of the numbers (800 numbers) for the train set
# train_numbers = random.sample(numbers, int(0.8 * len(numbers)))
# train_numbers.sort()
#
# # the rest numbers (200 numbers) are for the test set
# test_numbers = [n for n in numbers if n not in train_numbers]

train_filename = "/scratch/zczqyc4/360-videos-grouped/sep_small_trainlist.txt"
val_filename = "/scratch/zczqyc4/360-videos-grouped/sep_small_testlist.txt"
test_filename = "/scratch/zczqyc4/360-videos-grouped/sep_for_testlist.txt"

# write train numbers to the train file
with open(train_filename, "w") as file:
    for i in range(0, 20):
        class_name = classes[i]
        for j in range(1, 401):
            line = f"{class_name}/group{j}\n"
            file.write(line)

    # for i in train_numbers:
    #     line = f"AirShow/group{i}\n"
    #     file.write(line)

# write test numbers to the test file
with open(val_filename, "w") as file:
    for i in range(0, 20):
        class_name = classes[i]
        for j in range(401, 501):
            line = f"{class_name}/group{j}\n"
            file.write(line)
    # for i in test_numbers:
    #     line = f"AirShow/group{i}\n"
    #     file.write(line)
with open(test_filename, "w") as file:
    for i in range(0, 1):
        class_name = classes[i]
        for j in range(1, 17):
            line = f"{class_name}/group{j}\n"
            file.write(line)

