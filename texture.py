from http import server
import json
import os
import random
import socketserver
from PIL import Image
from PIL.ExifTags import TAGS

# List all files in single dir
def list_all_files(directory):
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

def get_list(question):
    # create 10 random integer numbers between 12 and 126, not using numpy
    random_integers = [random.randint(12, 126) for i in range(10)]

    # sort the list of numbers by the reverse of the string representation of the numbers
    random_integers.sort(key=lambda x: str(x)[::-1])



def add_material(textures, material):
    # trim and turn material into proper case
    material = material.strip().title()

    # return if material is empty
    if material == "":
        return

    if not material in textures:
        textures[material] = {
            "name": material,
            "files": [],
            "prompt": "",
            "negative_prompt": "",
            "tags": [],
            "categories": [],
            "status": "new"
        }

    return material

def import_texture(filename, textures):
    # get the name
    name = os.path.splitext(filename)[0].rstrip('0123456789 ')

    name = add_material(textures, name)

    # if the name is in the dictionary, check if the image is already in the list
    for texture in textures[name]["files"]:
        if texture["path"] == filename:
            # if the image is already in the list, break the loop
            break
    else:
        print("Adding " + filename)
        # load the filename as image with PIL
        img = Image.open("outputs/Textures/" + filename)
        # get the size of the image
        width, height = img.size
        exif_data = img.getexif()
        comment = exif_data[0x9286]
        data = json.loads(comment)

        # if the image is not in the list, add it to the list
        textures[name]["files"].append({
            "path": filename,
            "width": width,
            "height": height,
            "prompt": data["Prompt"],
            "negative_prompt": data["Negative Prompt"],
            "cfg": data["CFG"],
            "steps": data["Steps"],
            "material": data["Material"]
        })

def load_textures():
    # read the json file "textures.json" and load it into a dictionary, if it doesn't exist, create an empty dictionary
    try:
        with open('textures.json', 'r') as f:
            textures = json.load(f)
    except:
        try:
            with open('textures_backup.json', 'r') as f:
                textures = json.load(f)
        except:
            textures = {}
    return textures

def save_textures(textures):
    # write textures.json
    print("Saving textures.json")
    with open('textures_backup.json', 'w') as f:
        f.write(json.dumps(textures))
    with open('textures.json', 'w') as f:
        f.write(json.dumps(textures))

    # remove backup
    os.remove('textures_backup.json')


def import_textures():
    # read the json file "textures.json" and load it into a dictionary, if it doesn't exist, create an empty dictionary
    try:
        with open('textures.json', 'r') as f:
            textures = json.load(f)
    except:
        textures = {}

    for filename in list_all_files("outputs/Textures"):
        import_texture(filename, textures)

    save_textures(textures)

    # write textures dictionary to file
    with open('textures.json', 'w') as f:
        f.write(json.dumps(textures))


def clean_textures(textures):
    # remove files that don't exist
    for name in textures:
        for texture in list(textures[name]["files"]):
            if not os.path.isfile("outputs/Textures/" + texture["path"]):
                print("Removing " + texture["path"])
                textures[name]["files"].remove(texture)

    # remove textures that have no files
    for name in list(textures.keys()):
        if len(textures[name]["files"]) == 0:
            print("Removing " + name)
            del textures[name]


class WebHandler(server.SimpleHTTPRequestHandler):
    textures = {}
    def init() -> None:
        # load textures.json
        print("Loading textures.json")
        WebHandler.textures = load_textures()

    def save() -> None:
        # write textures.json
        print("Saving textures.json")
        with open('textures_backup.json', 'w') as f:
            f.write(json.dumps(WebHandler.textures))
        with open('textures.json', 'w') as f:
            f.write(json.dumps(WebHandler.textures))
        # remove backup
        os.remove('textures_backup.json')

    def do_GET(self):
        textures = WebHandler.textures
        if self.path == '/':
            self.path = 'index.html'

        if self.path == '/textures.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.textures).encode())
            return

        if self.path == '/next_material':
            print("Getting materials that aren't bad")
            materials = [name for name in textures if textures[name]["status"] != "bad"]
            print("Sorting materials by number of files")
            materials.sort(key=lambda name: len(textures[name]["files"]))
            self.send_response(200)
            self.send_header('Content-type', 'application/text')
            self.end_headers()
            self.wfile.write(materials[0].encode())
            return

        if self.path.startswith('/Textures/'):
            self.path = "/outputs" + self.path
        else:
            self.path = "www/" + self.path
        return server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        textures = WebHandler.textures
        if self.path == "/update":
            # get the body as json
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # load the json into a dictionary
            update = json.loads(body)
            print(update)

            if update["action"] == "good":
                textures[update["name"]]["status"] = "good"
            if update["action"] == "bad":
                textures[update["name"]]["status"] = "bad"
            if update["action"] == "new":
                textures[update["name"]]["status"] = "new"

            WebHandler.save()

        if self.path == "/add_materials":
            # get the body as json
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # split the string by linebreaks into a list
            materials = body.decode().splitlines()
            print(materials)

            for material in materials:
                add_material(textures, material)

            WebHandler.save()
            
        if self.path == "/add_images":
            # get the body as json
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # read images from body as json list
            images = json.loads(body)

            for image in images:
                import_texture(image, textures)

            WebHandler.save()

        self.send_response(200)
        self.send_header('Content-type', 'application/text')
        self.end_headers()
        self.wfile.write("OK".encode())

def run_server():
    import_textures()
    WebHandler.init()
    clean_textures(WebHandler.textures)
    WebHandler.save()
    socketserver.TCPServer(("", 8000), WebHandler).serve_forever()

def main():
    run_server()

if __name__ == "__main__":
    main()