import os

from numpy import full


root = "D:\\texturesSorted"

number = 1
last = ""

longest = 0

# List all directories in the directory
for directory in os.listdir(root):
    # List all files in the directory
    for file in os.listdir(os.path.join(root, directory)):
        # get the filename length
        length = len(file)
        # if longer than longest, update
        if length > longest:
            longest = length
            print(longest, file)

        # # Check there is no dash in the filename and it contains numbers
        # if "-" not in file and any(char.isdigit() for char in file):
        #     # create a new filename, remove the numbers
        #     new_filename = directory + " - " + file.translate({ord(i): None for i in '0123456789'}).replace(" .png", "")
        #     if (new_filename == last):
        #         number += 1
        #     else:
        #         number = 1
        #     last = new_filename

        #     while (os.path.exists(os.path.join(root, directory, new_filename + " " + str(number) + ".png"))):
        #         number += 1

        #     full_filename = new_filename + " " + str(number) + ".png"

        #     print(full_filename)

        #     # rename the file
        #     os.rename(os.path.join(root, directory, file), os.path.join(root, directory, full_filename))


            
            
        
