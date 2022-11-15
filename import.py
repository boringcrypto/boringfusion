import importlib
import glob, re, os
from core.data import StableDiffusionData
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
            data = StableDiffusionData()
            data.load_checkpoint(filename)
            print(data)
            print()
            data.save_native()

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
            data.save_native()

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
    
    for diffusers_directory in glob.glob("import/*"):
        if os.path.isdir(diffusers_directory):
            print(diffusers_directory)
            importer.import_diffusers(diffusers_directory)

    for checkpoint_filename in glob.glob("import/*.ckpt"):
        print(checkpoint_filename)
        importer.import_checkpoint(checkpoint_filename)

    # Update the model enum
    importlib.reload(model_map)
    importer.map = model_map.map

    importer.create_model_enums()


if __name__ == "__main__":
    main()

