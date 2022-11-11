import importlib
import glob, re, os
from core.data import StableDiffusionData

def main():
    import model_map

    for diffusers_directory in glob.glob("import/diffusers/*"):
        found = [
            os.path.exists(model.filename)
            for model in model_map.map.values() 
            if os.path.basename(os.path.normpath(diffusers_directory)) in model.found_in
        ]

        if not len(found) or not all(found):
            data = StableDiffusionData()
            data.load_diffusers(diffusers_directory)
            print(data)
            print()
            data.save_native()

    for checkpoint_filename in glob.glob("import/checkpoints/*.ckpt"):
        found = [
            os.path.exists(model.filename)
            for model in model_map.map.values() 
            if os.path.split(checkpoint_filename)[1] in model.found_in
        ]

        if not len(found) or not all(found):
            data = StableDiffusionData()
            data.load_checkpoint(checkpoint_filename)
            print(data)
            print()
            data.save_native()

    # Update the model enum
    importlib.reload(model_map)

    lines = [
        "from enum import Enum",
    ]

    def add_model(name, models):
        lines.append("")
        lines.append(f"class {name}(Enum):")
        lines.extend(
            [
                "    " + re.sub(r'[^a-zA-Z0-9]', '_', model.shortname or model.name) + ' = "' + key + '"' 
                for key, model in model_map.map.items() if model.model in models and os.path.exists(model.filename)
            ])

    add_model("unet", ["UNet", "UNet EMA"])
    add_model("clip", ["CLIP Encoder"])
    add_model("decoder", ["VAE Decoder"])
    add_model("encoder", ["VAE Encoder"])

    with open('models.py', 'w') as f:
        f.write(str.join("\n", lines))

if __name__ == "__main__":
    main()

