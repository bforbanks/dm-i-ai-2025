TOTAL_BRIGHT_SCORE = 0.01858123972090343 #when we predict true on every single pixel

# possible_naevnere = []
# for i in range(100000000):
#     if 0.00007652258988783675*i % 1 < 1e-10:
#         print(i)
#         possible_naevnere.append(i)

# print(possible_naevnere)

# possible_naevnere = []
# for i in range(100000000):
#     if 0.01858123972090343*i % 1 < 1e-7:
#         if round(i * 0.01858123972090343) % 2 == 0:
#             print(i)
#             possible_naevnere.append(i)

# print(possible_naevnere)

import csv
import os
import time

CSV_PATH = os.path.join('validation', 'results.csv')

total_pixels = 0

with open(CSV_PATH, 'r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['total_number_of_pixels']:  # check if the value exists
            total_pixels += int(row['total_number_of_pixels'])

print(f"Total number of pixels across all images: {total_pixels:,}")


# count_total_pixels()
#28619000 
#26049600 total number pixel
#619,574 number true positives?

# img = 1
# number = 0.000028293951212051693*129 #img1
# number = 0.00007652258988783675*129 #img2 0.00007652303553021612
# number = 0.00006783929804606573*129 #img3
# # number = 0.01858123972090343 #total

# possible_naevnere = []
# for img in range(1,4):
#     with open(CSV_PATH, 'r', newline='') as file:
#         reader = csv.DictReader(file)
#         #choose row number img
#         for i, row in enumerate(reader):
#             if row['picture_id'] == str(img):
#                 total_pixels = int(row['total_number_of_pixels'])
#                 full_bright_score = float(row['full_bright_score'])*129
#                 print(full_bright_score,total_pixels)
#                 break

#     possible_naevnere.append([])
#     for naevner in range(1,total_pixels*2+1):
#         # print(naevner)
#         if full_bright_score*naevner % 1 < 1e-5:
#             print(full_bright_score*naevner)
#             if round(naevner * full_bright_score) % 2 == 0:
#                 taeller = round(naevner*full_bright_score)
#                 TP = taeller/2
#                 FP = naevner - taeller
#                 print("testing corectness",naevner,2*TP+FP, TP+FP, total_pixels)
#                 time.sleep(1)
#                 if TP + FP == total_pixels:
#                     print("Wuhuuuu, naevner found",naevner, TP, FP,total_pixels)
#                 possible_naevnere[img-1].append(i)

    

    # print(possible_naevnere)