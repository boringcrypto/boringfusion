import os
import glob
import hashlib
from collections import OrderedDict

import torch
from core import diffusers_mappings

def treeify(items):
    tree = {}

    for item in items:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})
    
    return tree

class ModelLayers(OrderedDict):
    def __init__(self, name, layer_count) -> None:
        self.name = name
        self.layer_count = layer_count
        super().__init__()

    @property
    def dtypes(self):
        types = [str(layer.dtype) for layer in list(self.values())]
        return str.join(", ", [str(dtype) for dtype in set(types)])

    @property
    def hash(self):
        return hashlib.md5(str(self.items()).encode()).hexdigest()[:8]

    @property
    def parameter_count(self):
        return sum(torch.numel(layer) for layer in self.values())

    def set(self, layers):
        if len(layers) == self.layer_count:
            self.clear()
            self.update(layers)
        else:
            print(self.name, "model has the wrong number of layers, skipped", len(layers), self.layer_count)

    def set_from_state_dict(self, state_dict, base_key="", mapping=None):
        base_key = base_key if base_key.endswith(".") or not len(base_key) else base_key + "."
        keys = [key for key in state_dict.keys() if not len(base_key) or key.startswith(base_key)]
        layers = {mapping[key[len(base_key):]] if mapping else key[len(base_key):] : state_dict[key] for key in keys}
        
        # Sort by key to ensure order is always the same to generate the same hash
        self.set({key : layers[key] for key in sorted(layers.keys())})

    def half(self):
        for key, layer in self.items():
            self[key] = layer.half()

    def __str__(self):
        if len(self):
            return self.name + ": " + self.hash + " - " + self.dtypes + " - " + str(self.parameter_count // 1000000) + "M params"
        else:
            return self.name + ": None"

class StableDiffusionModelData:
    def __init__(self) -> None:
        self.filename = ""
        self.clip_text_encoder_layers = ModelLayers("CLIP Encoder", 197)
        self.vae_encoder_layers = ModelLayers("VAE Encoder", 106)
        self.vae_decoder_layers = ModelLayers("VAE Decoder", 138)
        self.unet_layers = ModelLayers("UNet", 686)

    def load_checkpoint(self, filename):
        self.filename = filename
        checkpoint_data = torch.load(filename, map_location="cpu")
        state_dict = checkpoint_data["state_dict"] if 'state_dict' in checkpoint_data.keys() else checkpoint_data
        key_tree = treeify(state_dict.keys())

        if "cond_stage_model" in key_tree:
            # The Novel AI ckpt skips the text_model part
            clip_base_key = "cond_stage_model.transformer.text_model" if "text_model" in key_tree["cond_stage_model"]["transformer"] else "cond_stage_model.transformer"
            self.clip_text_encoder_layers.set_from_state_dict(state_dict, clip_base_key)

        if "first_stage_model" in key_tree:
            if "encoder" in key_tree["first_stage_model"]:
                self.vae_encoder_layers.set_from_state_dict(state_dict, "first_stage_model.encoder")

            if "decoder" in key_tree["first_stage_model"]:
                self.vae_decoder_layers.set_from_state_dict(state_dict, "first_stage_model.decoder")

        if "model" in key_tree and "diffusion_model" in key_tree["model"]:
            self.unet_layers.set_from_state_dict(state_dict, "model.diffusion_model")

        return self

    def load_diffusers(self, directory):
        self.filename = directory

        clip_text_encoder_layers = torch.load(os.path.join(directory, "text_encoder", "pytorch_model.bin"), map_location='cpu')
        self.clip_text_encoder_layers.set_from_state_dict(clip_text_encoder_layers, "text_model")

        vae_layers = torch.load(os.path.join(directory, "vae", "diffusion_pytorch_model.bin"), map_location='cpu')

        self.vae_decoder_layers.set_from_state_dict(vae_layers, "decoder", diffusers_mappings.vae_encoder)
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_decoder_layers['mid.attn_1.k.weight'] = self.vae_decoder_layers['mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.proj_out.weight'] = self.vae_decoder_layers['mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.q.weight'] = self.vae_decoder_layers['mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.v.weight'] = self.vae_decoder_layers['mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        self.vae_encoder_layers.set_from_state_dict(vae_layers, "encoder", diffusers_mappings.vae_decoder)
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_encoder_layers['mid.attn_1.k.weight'] = self.vae_encoder_layers['mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.proj_out.weight'] = self.vae_encoder_layers['mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.q.weight'] = self.vae_encoder_layers['mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.v.weight'] = self.vae_encoder_layers['mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        unet_layers = torch.load(os.path.join(directory, "unet", "diffusion_pytorch_model.bin"), map_location='cpu')
        self.unet_layers.set_from_state_dict(unet_layers, "", diffusers_mappings.unet)

        return self

    @property
    def full_state_dict(self):
        full = {
            "cond_stage_model": {},
            "first_stage_model": {},
            "model": {}
        }
        for key in self.clip_text_encoder_layers.keys():
            full["cond_stage_model"]["transformer.text_model." + key] = self.clip_text_encoder_layers[key]

        for key in self.vae_encoder_layers.keys():
            full["first_stage_model"]["encoder." + key] = self.vae_encoder_layers[key]

        for key in self.vae_decoder_layers.keys():
            full["first_stage_model"]["decoder." + key] = self.vae_decoder_layers[key]

        for key in self.unet_layers.keys():
            full["model"]["diffusion_model." + key] = self.unet_layers[key]

        return full


    def __str__(self) -> str:
        lines = []
        lines.append("File " + self.filename)
        lines.append(str(self.clip_text_encoder_layers))
        lines.append(str(self.vae_encoder_layers))
        lines.append(str(self.vae_decoder_layers))
        lines.append(str(self.unet_layers))
        return str.join("\r\n", lines)

def main():
    # base = StableDiffusionModelData().load_checkpoint("data/checkpoints/sd-v1-4.ckpt")
    # print(base)
    # print()

    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    with torch.autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]  
        
    image.save("astronaut_rides_horse.png")    

    # wd = StableDiffusionModelData().load_checkpoint("data/checkpoints/LOKEAN_MISSIONARY_POV.ckpt")
    # wd.unet_layers.half()
    # print(wd)

    # diff = ModelLayers("Diff", 686)
    # for key in base.unet_layers.keys():
    #     diff[key] = base.unet_layers[key] - wd.unet_layers[key]

    # torch.save(wd.unet_layers, "full.bin")
    # torch.save(diff, "diff.bin")
    #print(diff)

    # for checkpoint_filename in glob.glob("data/checkpoints/*.ckpt"):
    #     data = StableDiffusionModelData()
    #     data.load_checkpoint(checkpoint_filename)
    #     print(data)

if __name__ == "__main__":
    main()

