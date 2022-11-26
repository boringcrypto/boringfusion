import importlib
import glob, re, os
from core.data.sd_data import StableDiffusionData, StableDiffusionDataImporter
from core.safe_unpickler import torch_load
import model_map


class Importer():
    def __init__(self, map) -> None:
        self.map = map

    def import_checkpoint(self, filename):
        found = [
            os.path.exists(model.filename)
            for model in self.map.values() 
            if os.path.split(filename)[1] in model.found_in
        ]

        if not len(found) or not all(found):
            data = StableDiffusionDataImporter()
            data.import_checkpoint(filename)
            print(data)
            print()
            self.save_map()

    def import_diffusers(self, directory):
        found = [
            os.path.exists(model.filename)
            for model in self.map.values() 
            if os.path.basename(os.path.normpath(directory)) in model.found_in
        ]

        if not len(found) or not all(found):
            data = StableDiffusionData()
            data.load_diffusers(directory)
            print(data)
            print()
            data.save_native(self.map)
            self.save_map()

    def import_hypernet(self, filename):
        found = [
            os.path.exists(model.filename)
            for model in self.map.values() 
            if os.path.split(filename)[1] in model.found_in
        ]

        if not len(found) or not all(found):
            data = StableDiffusionData()
            data.load_hypernet(filename)
            print(data)
            print()
            data.save_native(self.map)

    def import_file(self, filename):
        found = [
            os.path.exists(model.filename)
            for model in self.map.values() 
            if os.path.split(filename)[1] in model.found_in
        ]

        if not len(found) or not all(found):
            data = torch_load(filename)
            print(filename, data.keys())

    def save_map(self):
        full_module = 'from core.data import ModelMap, ModelDataInfo\n\nmap = ' + str(self.map)

        # Saving a backup first, just in case the save gets interupted halfway through
        with open('model_map_backup.py', 'w') as f:
            f.write(full_module)

        with open('model_map.py', 'w') as f:
            f.write(full_module)

        # Remove the backup after a succesful save
        os.remove('model_map_backup.py')


    def create_model_enums(self):
        lines = [
            "from enum import Enum",
        ]

        def add_model(name, models):
            lines.append("")
            lines.append(f"class {name}(Enum):")
            lines.extend(
                [
                    "    " + re.sub(r'[^a-zA-Z0-9]', '_', model.shortname or model.name) + ' = "' + key + '"' 
                    for key, model in self.map.items() if model.model in models and os.path.exists(model.filename)
                ])

        add_model("unet", ["UNet", "UNet EMA"])
        add_model("clip", ["CLIP Encoder"])
        add_model("decoder", ["VAE Decoder"])
        add_model("encoder", ["VAE Encoder"])

        with open('models.py', 'w') as f:
            f.write(str.join("\n", lines))


def main():
    importer = Importer(model_map.map)

    for path in glob.glob("import/*"):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
            importer.import_diffusers(path)
        # else:
        #     importer.import_file(path)

    for checkpoint_filename in glob.glob("import/*.ckpt"):
        print(checkpoint_filename)
        importer.import_checkpoint(checkpoint_filename)

    # Update the model enum
    importlib.reload(model_map)
    importer.map = model_map.map

    importer.create_model_enums()


if __name__ == "__main__":
    main()

