import http
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





        

    




def import_textures():
    # read the json file "textures.json" and load it into a dictionary, if it doesn't exist, create an empty dictionary
    try:
        with open('textures.json', 'r') as f:
            textures = json.load(f)
    except:
        textures = {}

    for filename in list_all_files("outputs/Textures"):

        # get the name
        name = os.path.splitext(filename)[0].rstrip('0123456789 ')

        if not name in textures:
            textures[name] = {
                "name": name,
                "files": [],
                "prompt": "",
                "negative_prompt": "",
                "tags": [],
                "categories": [],
                "status": "new"
            }

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
        
    # write textures dictionary to file
    with open('textures.json', 'w') as f:
        f.write(json.dumps(textures))


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    textures = {}
    def init() -> None:
        # load textures.json
        print("Loading textures.json")
        with open('textures.json', 'r') as f:
            MyHttpRequestHandler.textures = json.load(f)

    def save() -> None:
        # write textures.json
        print("Saving textures.json")
        with open('textures_backup.json', 'w') as f:
            f.write(json.dumps(MyHttpRequestHandler.textures))
        with open('textures.json', 'w') as f:
            f.write(json.dumps(MyHttpRequestHandler.textures))
        # remove backup
        os.remove('textures_backup.json')

    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'

        if self.path == '/textures.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.textures).encode())
            return

        if self.path.startswith('/Textures/'):
            self.path = "/outputs" + self.path
        else:
            self.path = "www/" + self.path
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == "/update":
            # get the body as json
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            # load the json into a dictionary
            update = json.loads(body)
            print(update)

            if update["action"] == "good":
                self.textures[update["name"]]["status"] = "good"
            if update["action"] == "bad":
                self.textures[update["name"]]["status"] = "bad"
            if update["action"] == "new":
                self.textures[update["name"]]["status"] = "new"

            MyHttpRequestHandler.save()
            

def main():
    # import_textures()
    MyHttpRequestHandler.init()

    # os.system("start http://localhost:8000")

    socketserver.TCPServer(("", 8000), MyHttpRequestHandler).serve_forever()

if __name__ == "__main__":
    main()