import os
from PIL import Image

# List all files in single dir
def list_all_files(directory):
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

def main():
    # create an empty dictionary
    textures = {}
    for file in list_all_files("outputs/Textures"):
        filename = file
        # load the filename as image with PIL
        img = Image.open("outputs/Textures/" + filename)
        # get the size of the image
        width, height = img.size

        #strip the extension
        file = os.path.splitext(file)[0]
        # remove the postfix numeric characters
        file = file.rstrip('0123456789 ')
        # add or extend file to the dictionary as key and a list of filename as value
        textures.setdefault(file, []).append({
            "path": filename,
            "width": width,
            "height": height
        })
        
    print(textures)
    

if __name__ == "__main__":
    main()