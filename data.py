import os
import glob
import hashlib
import core.safe_unpickler
from collections import OrderedDict

import torch
from core import diffusers_mappings
from core.safe_unpickler import torch_load

def treeify(items):
    tree = {}

    for item in items:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})
    
    return tree

class ModelLayers(OrderedDict):
    def __init__(self, name="", layer_count=0) -> None:
        self.name = name
        self.layer_count = layer_count
        super().__init__()

    @property
    def dtypes(self):
        types = [str(layer.dtype) for layer in list(self.values())]
        return str.join(", ", [str(dtype) for dtype in set(types)])

    @property
    def version_hash(self):
        total = torch.zeros(256)
        for key in self.keys():
            flat = self[key].flatten()
            length = flat.size(0)
            if (length<=256):
                total[0:length] += flat
            else:
                total.add_(flat[length // 2 - 128 : length // 2 + 128])
        hash = hashlib.md5(str([e.item() for e in total]).encode()).hexdigest()[:8]

        return hash

    @property
    def content_hash(self):
        total = torch.zeros(256)
        for key in self.keys():
            flat = self[key].flatten()
            length = flat.size(0)
            if (length<=256):
                total[0:length] += flat.half()
            else:
                total.add_(flat[length // 2 - 128 : length // 2 + 128].half())
        hash = hashlib.md5(str([e.item() for e in total]).encode()).hexdigest()[:8]

        return hash

    @property
    def parameter_count(self):
        return sum(torch.numel(layer) for layer in self.values())

    def set(self, layers):
        if len(layers) == self.layer_count:
            self.clear()
            self.update(layers)
        else:
            print(self.name, "model has the wrong number of layers, skipped", len(layers), self.layer_count)

    def set_from_state_dict(self, state_dict, base_key="", mapping=None, prepend=""):
        keys = [key for key in state_dict.keys() if not len(base_key) or key.startswith(base_key)]
        layers = {prepend + mapping[key[len(base_key):]] if mapping else prepend + key[len(base_key):] : state_dict[key] for key in keys}
        
        # Sort by key to ensure order is always the same to generate the same hash
        self.set({key : layers[key] for key in sorted(layers.keys())})

    def half_(self):
        """Converts all layers to half precision (fp16)
        """
        for key in self.keys():
            self[key] = self[key].half()

    def save(self, filename):
        torch.save({
            "name": self.name,
            "layer_count": self.layer_count,
            "state_dict": { key:value for key, value in self.items() },

            # This meta-data is stored so we don't have to load the full data to get this info
            # Once loaded, this info should be ignored and the actual class properties used
            "content_hash": self.content_hash,
            "version_hash": self.version_hash,
            "dtypes": self.dtypes,
            "parameter_count": self.parameter_count
        }, filename)

    @classmethod
    def load(cls, filename):
        data = torch_load(filename, map_location="cpu")
        model_layers = cls(data["name"], data["layer_count"])
        model_layers.set_from_state_dict(data["state_dict"])

        return model_layers

    def merge_(self, model, weight=0.5):
        """Merge with another model (in place)

        Args:
            model (dict or ModelLayers): Another model with the same keys and shapes
            weight (float, optional): The weight of the model being merged in. A weight of one doesn't do anything, weight of 1 overwrites with the model. Defaults to 0.5.
        """
        for key in self.keys():
            self[key] = self[key] * (1 - weight) + model[key] * weight

        return self

    def merge(self, model, weight=0.5):
        """Merge with another model and returns the merged ModelLayers

        Args:
            model (dict or ModelLayers): Another model with the same keys and shapes
            weight (float, optional): The weight of the model being merged in. A weight of one doesn't do anything, weight of 1 overwrites with the model. Defaults to 0.5.
        """
        combined = ModelLayers(self.name, self.layer_count)
        for key in self.keys():
            combined[key] = self[key] * (1 - weight) + model[key] * weight
        
        return combined

    def add_diff_(self, base_model, trained_model, weight=1):
        for key in self.keys():
            self[key] = self[key] + (trained_model[key] - base_model[key]) * weight

        return self

    def add_diff_(self, base_model, trained_model, weight=1):
        combined = ModelLayers(self.name, self.layer_count)
        for key in self.keys():
            combined[key] = self[key] + (trained_model[key] - base_model[key]) * weight

        return combined

    def __str__(self):
        if len(self):
            return self.name + ": " + self.content_hash + ":" + self.version_hash + " - " + self.dtypes + " - " + str(self.parameter_count // 1000000) + "M params"
        else:
            return self.name + ": None"

class StableDiffusionModelData:
    def __init__(self) -> None:
        self.filename = ""
        self.clip_text_encoder_layers = ModelLayers("CLIP Encoder", 197)
        self.vae_encoder_layers = ModelLayers("VAE Encoder", 106)
        self.vae_decoder_layers = ModelLayers("VAE Decoder", 138)
        self.unet_layers = ModelLayers("UNet", 686)
        self.unet_ema_layers = ModelLayers("UNet EMA", 686)

    def load_checkpoint(self, filename, unsafe=False):
        self.filename = filename

        checkpoint_data = torch_load(filename)

        state_dict = checkpoint_data["state_dict"] if 'state_dict' in checkpoint_data.keys() else checkpoint_data
        key_tree = treeify(state_dict.keys())

        if "cond_stage_model" in key_tree:
            # The Novel AI ckpt skips the text_model part
            clip_base_key = "cond_stage_model.transformer.text_model." if "text_model" in key_tree["cond_stage_model"]["transformer"] else "cond_stage_model.transformer."
            self.clip_text_encoder_layers.set_from_state_dict(state_dict, clip_base_key, prepend="text_model.")

        if "first_stage_model" in key_tree:
            if "encoder" in key_tree["first_stage_model"]:
                self.vae_encoder_layers.set_from_state_dict(state_dict, "first_stage_model.encoder.", prepend="encoder")
                self.vae_encoder_layers["quant_conv.bias"] = state_dict["first_stage_model.quant_conv.bias"]
                self.vae_encoder_layers["quant_conv.weight"] = state_dict["first_stage_model.quant_conv.weight"]

            if "decoder" in key_tree["first_stage_model"]:
                self.vae_decoder_layers.set_from_state_dict(state_dict, "first_stage_model.decoder.", prepend="decoder.")
                self.vae_decoder_layers["post_quant_conv.bias"] = state_dict["first_stage_model.post_quant_conv.bias"]
                self.vae_decoder_layers["post_quant_conv.weight"] = state_dict["first_stage_model.post_quant_conv.weight"]

        if "model" in key_tree and "diffusion_model" in key_tree["model"]:
            self.unet_layers.set_from_state_dict(state_dict, "model.diffusion_model.")

        if "model_ema" in key_tree and len(key_tree["model_ema"]) >= self.unet_ema_layers.layer_count:
            key_map = { key.replace(".", "") : key for key in self.unet_layers.keys() }
            self.unet_ema_layers.set_from_state_dict(state_dict, "model_ema.diffusion_model", key_map)

        return self

    def load_diffusers(self, directory):
        self.filename = directory

        clip_text_encoder_layers = torch_load(os.path.join(directory, "text_encoder", "pytorch_model.bin"), map_location='cpu')
        self.clip_text_encoder_layers.set_from_state_dict(clip_text_encoder_layers, "text_model.")

        vae_layers = torch_load(os.path.join(directory, "vae", "diffusion_pytorch_model.bin"), map_location='cpu')

        self.vae_decoder_layers.set_from_state_dict(vae_layers, "decoder.", diffusers_mappings.vae_encoder)
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_decoder_layers['mid.attn_1.k.weight'] = self.vae_decoder_layers['mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.proj_out.weight'] = self.vae_decoder_layers['mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.q.weight'] = self.vae_decoder_layers['mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['mid.attn_1.v.weight'] = self.vae_decoder_layers['mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        self.vae_encoder_layers.set_from_state_dict(vae_layers, "encoder.", diffusers_mappings.vae_decoder)
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_encoder_layers['mid.attn_1.k.weight'] = self.vae_encoder_layers['mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.proj_out.weight'] = self.vae_encoder_layers['mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.q.weight'] = self.vae_encoder_layers['mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['mid.attn_1.v.weight'] = self.vae_encoder_layers['mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        unet_layers = torch_load(os.path.join(directory, "unet", "diffusion_pytorch_model.bin"), map_location='cpu')
        self.unet_layers.set_from_state_dict(unet_layers, "", diffusers_mappings.unet)

        return self

    def save_native(self):
        # if len(self.clip_text_encoder_layers):
        #     filename = "data/clip-text-encoder/" + self.clip_text_encoder_layers.content_hash + "-" + self.clip_text_encoder_layers.version_hash + "-" + self.clip_text_encoder_layers.dtypes + ".bin"
        #     if not os.path.exists(filename):
        #         self.clip_text_encoder_layers.save(filename)

        if len(self.vae_encoder_layers):
            filename = "data/vae-encoder/" + self.vae_encoder_layers.content_hash + "-" + self.vae_encoder_layers.version_hash + "-" + self.vae_encoder_layers.dtypes + ".bin"
            if not os.path.exists(filename):
                self.vae_encoder_layers.save(filename)

        if len(self.vae_decoder_layers):
            filename = "data/vae-decoder/" + self.vae_decoder_layers.content_hash + "-" + self.vae_decoder_layers.version_hash + "-" + self.vae_decoder_layers.dtypes + ".bin"
            if not os.path.exists(filename):
                self.vae_decoder_layers.save(filename)
            # print('"' + os.path.split(self.filename)[1][0:-5] + '": "' + filename + '",')

        if len(self.unet_ema_layers):
            filename = "data/unet/" + self.unet_ema_layers.content_hash + "-" + self.unet_ema_layers.version_hash + "-" + self.unet_ema_layers.dtypes + ".bin"
            if not os.path.exists(filename):
                self.unet_ema_layers.save(filename)
            # print('"' + os.path.split(self.filename)[1][0:-5] + '": "' + filename + '",')
        elif len(self.unet_layers):
            filename = "data/unet/" + self.unet_layers.content_hash + "-" + self.unet_layers.version_hash + "-" + self.unet_layers.dtypes + ".bin"
            if not os.path.exists(filename):
                self.unet_layers.save(filename)
            print('"' + os.path.split(self.filename)[1][0:-5] + '": "' + filename + '",')


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
        lines.append(str(self.unet_ema_layers))
        return str.join("\r\n", lines)

def main():
    # base = StableDiffusionModelData().load_checkpoint("data/checkpoints/sd-v1-4.ckpt")
    # print(base)
    # print()

    # wd = StableDiffusionModelData().load_checkpoint("data/checkpoints/v1-5-pruned.ckpt")
    # print(wd)

    # diff = ModelLayers("Diff", 686)
    # for key in base.unet_layers.keys():
    #     diff[key] = base.unet_layers[key] - wd.unet_layers[key]

    # torch.save(wd.unet_layers, "full.bin")
    # torch.save(diff, "diff.bin")
    #print(diff)

    for checkpoint_filename in glob.glob("import/checkpoints/*.ckpt"):
        data = StableDiffusionModelData()
        data.load_checkpoint(checkpoint_filename)
        try:

            # print(data)
            # print()
            data.save_native()
        except Exception as exception:
            print(checkpoint_filename, " cannot be loaded safely", exception)

if __name__ == "__main__":
    main()

