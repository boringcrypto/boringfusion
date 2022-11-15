import os
import torch
from core import diffusers_mappings
from core.safe_unpickler import torch_load
from .model_data import ModelData
try:
    from model_map_backup import map
except:
    from model_map import map

def treeify(items):
    tree = {}

    for item in items:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})
    
    return tree


class StableDiffusionData:
    def __init__(self) -> None:
        self.filename = ""
        self.clip_text_encoder_layers = ModelData("CLIP Encoder", 197)
        self.vae_encoder_layers = ModelData("VAE Encoder", 106)
        self.vae_decoder_layers = ModelData("VAE Decoder", 138)
        self.unet_layers = ModelData("UNet", 686)
        self.unet_ema_layers = ModelData("UNet EMA", 686)

    def load_checkpoint(self, filename, unsafe=False):
        self.filename = os.path.split(filename)[1]

        checkpoint_data = torch_load(filename, map_location='cpu')

        state_dict = checkpoint_data["state_dict"] if 'state_dict' in checkpoint_data.keys() else checkpoint_data
        key_tree = treeify(state_dict.keys())

        if "cond_stage_model" in key_tree:
            # The Novel AI ckpt skips the text_model part
            clip_base_key = "cond_stage_model.transformer.text_model." if "text_model" in key_tree["cond_stage_model"]["transformer"] else "cond_stage_model.transformer."
            self.clip_text_encoder_layers.set_from_state_dict(state_dict, clip_base_key, prepend="text_model.")

        if "first_stage_model" in key_tree:
            if "encoder" in key_tree["first_stage_model"]:
                self.vae_encoder_layers.set_from_state_dict(state_dict, "first_stage_model.encoder.", prepend="encoder.")
                self.vae_encoder_layers["quant_conv.bias"] = state_dict["first_stage_model.quant_conv.bias"]
                self.vae_encoder_layers["quant_conv.weight"] = state_dict["first_stage_model.quant_conv.weight"]
                self.vae_encoder_layers.layer_count += 2

            if "decoder" in key_tree["first_stage_model"]:
                self.vae_decoder_layers.set_from_state_dict(state_dict, "first_stage_model.decoder.", prepend="decoder.")
                self.vae_decoder_layers["post_quant_conv.bias"] = state_dict["first_stage_model.post_quant_conv.bias"]
                self.vae_decoder_layers["post_quant_conv.weight"] = state_dict["first_stage_model.post_quant_conv.weight"]
                self.vae_decoder_layers.layer_count += 2

        if "encoder" in key_tree and "quant_conv" in key_tree and "conv_in" in key_tree["encoder"]:
            self.vae_encoder_layers.set_from_state_dict(state_dict, "encoder.", prepend="encoder.")
            self.vae_encoder_layers["quant_conv.bias"] = state_dict["quant_conv.bias"]
            self.vae_encoder_layers["quant_conv.weight"] = state_dict["quant_conv.weight"]
            self.vae_encoder_layers.layer_count += 2

        if "decoder" in key_tree and "post_quant_conv" in key_tree and "conv_in" in key_tree["decoder"]:
            self.vae_decoder_layers.set_from_state_dict(state_dict, "decoder.", prepend="decoder.")
            self.vae_decoder_layers["post_quant_conv.bias"] = state_dict["post_quant_conv.bias"]
            self.vae_decoder_layers["post_quant_conv.weight"] = state_dict["post_quant_conv.weight"]
            self.vae_decoder_layers.layer_count += 2

        if "model" in key_tree and "diffusion_model" in key_tree["model"]:
            self.unet_layers.set_from_state_dict(state_dict, "model.diffusion_model.")

        if "model_ema" in key_tree and len(key_tree["model_ema"]) >= self.unet_ema_layers.layer_count:
            key_map = { key.replace(".", "") : key for key in self.unet_layers.keys() }
            self.unet_ema_layers.set_from_state_dict(state_dict, "model_ema.diffusion_model", key_map)

        return self

    def load_diffusers(self, directory):
        self.filename = os.path.basename(os.path.normpath(directory))

        clip_text_encoder_layers = torch_load(os.path.join(directory, "text_encoder", "pytorch_model.bin"), map_location='cpu')
        self.clip_text_encoder_layers.set_from_state_dict(clip_text_encoder_layers, "text_model.")

        vae_layers = torch_load(os.path.join(directory, "vae", "diffusion_pytorch_model.bin"), map_location='cpu')

        self.vae_decoder_layers.set_from_state_dict(vae_layers, "decoder.", diffusers_mappings.vae_decoder, "decoder.")
        self.vae_decoder_layers["post_quant_conv.bias"] = vae_layers["post_quant_conv.bias"]
        self.vae_decoder_layers["post_quant_conv.weight"] = vae_layers["post_quant_conv.weight"]
        self.vae_decoder_layers.layer_count += 2
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_decoder_layers['decoder.mid.attn_1.k.weight'] = self.vae_decoder_layers['decoder.mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['decoder.mid.attn_1.proj_out.weight'] = self.vae_decoder_layers['decoder.mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['decoder.mid.attn_1.q.weight'] = self.vae_decoder_layers['decoder.mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_decoder_layers['decoder.mid.attn_1.v.weight'] = self.vae_decoder_layers['decoder.mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        self.vae_encoder_layers.set_from_state_dict(vae_layers, "encoder.", diffusers_mappings.vae_encoder, "encoder.")
        self.vae_encoder_layers["quant_conv.bias"] = vae_layers["quant_conv.bias"]
        self.vae_encoder_layers["quant_conv.weight"] = vae_layers["quant_conv.weight"]
        self.vae_encoder_layers.layer_count += 2
        # These layers have the correct number of elements, but coem in a different shape
        self.vae_encoder_layers['encoder.mid.attn_1.k.weight'] = self.vae_encoder_layers['encoder.mid.attn_1.k.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['encoder.mid.attn_1.proj_out.weight'] = self.vae_encoder_layers['encoder.mid.attn_1.proj_out.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['encoder.mid.attn_1.q.weight'] = self.vae_encoder_layers['encoder.mid.attn_1.q.weight'].reshape(torch.Size([512, 512, 1, 1]))
        self.vae_encoder_layers['encoder.mid.attn_1.v.weight'] = self.vae_encoder_layers['encoder.mid.attn_1.v.weight'].reshape(torch.Size([512, 512, 1, 1]))

        unet_layers = torch_load(os.path.join(directory, "unet", "diffusion_pytorch_model.bin"), map_location='cpu')
        self.unet_layers.set_from_state_dict(unet_layers, "", diffusers_mappings.unet)

        return self

    def load_hypernet(self, filename):
        self.filename = os.path.split(filename)[1]

        data = torch_load(filename, map_location='cpu')
       
        if "state_dict" in data:
            data = data["state_dict"]

        key_tree = treeify(data.keys())

        if "encoder" in key_tree and "quant_conv" in key_tree and "conv_in" in key_tree["encoder"]:
            self.vae_encoder_layers.set_from_state_dict(data, "encoder.", prepend="encoder.")
            self.vae_encoder_layers["quant_conv.bias"] = data["quant_conv.bias"]
            self.vae_encoder_layers["quant_conv.weight"] = data["quant_conv.weight"]
            self.vae_encoder_layers.layer_count += 2

        if "decoder" in key_tree and "post_quant_conv" in key_tree and "conv_in" in key_tree["decoder"]:
            self.vae_decoder_layers.set_from_state_dict(data, "decoder.", prepend="decoder.")
            self.vae_decoder_layers["post_quant_conv.bias"] = data["post_quant_conv.bias"]
            self.vae_decoder_layers["post_quant_conv.weight"] = data["post_quant_conv.weight"]
            self.vae_decoder_layers.layer_count += 2

        # print(data.keys())

    def save_native(self):
        self.clip_text_encoder_layers.save_native(self.filename, "data/clip-text-encoder/", map)
        self.vae_encoder_layers.save_native(self.filename, "data/vae-encoder/", map)
        self.vae_decoder_layers.save_native(self.filename, "data/vae-decoder/", map)
        if not len(self.unet_ema_layers):
            self.unet_layers.save_native(self.filename, "data/unet/", map)
        self.unet_ema_layers.save_native(self.filename, "data/unet/", map)

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
        def add(layers):
            if len(layers):
                lines.append(str(layers))
        
        add(self.clip_text_encoder_layers)
        add(self.vae_encoder_layers)
        add(self.vae_decoder_layers)
        add(self.unet_layers)
        add(self.unet_ema_layers)
        
        return str.join("\r\n", lines)
