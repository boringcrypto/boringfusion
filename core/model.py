from modules.stable_diffusion import StableDiffusion
from enum import Enum
from transformers import logging

logging.set_verbosity_error()

class DiffusionModelInfo:
    def __init__(self, filename, name, info_url, download_url, description):
        self.filename = filename
        self.name = name
        self.info_url = info_url
        self.download_url = download_url
        self.description = description

class DIFF_MODEL(Enum):
    StableDiffusion1_1 = DiffusionModelInfo(
        "sd-v1-1",
        "Stable Diffusion v1.1", 
        "https://huggingface.co/CompVis/stable-diffusion-v-1-1-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\n The Stable-Diffusion-v-1-1 was trained on 237,000 steps at resolution 256x256 on laion2B-en, followed by 194,000 steps at resolution 512x512 on laion-high-resolution (170M examples from LAION-5B with resolution >= 1024x1024)."
    )
    StableDiffusion1_2 = DiffusionModelInfo(
        "sd-v1-2", 
        "Stable Diffusion v1.2", 
        "https://huggingface.co/CompVis/stable-diffusion-v-1-2-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\n The Stable-Diffusion-v-1-2 checkpoint was initialized with the weights of the Stable-Diffusion-v-1-1 checkpoint and subsequently fine-tuned on 515,000 steps at resolution 512x512 on \"laion-improved-aesthetics\" (a subset of laion2B-en, filtered to images with an original size >= 512x512, estimated aesthetics score > 5.0, and an estimated watermark probability < 0.5."
    )
    StableDiffusion1_3 = DiffusionModelInfo(
        "sd-v1-3", 
        "Stable Diffusion v1.3", 
        "https://huggingface.co/CompVis/stable-diffusion-v-1-3-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\n The Stable-Diffusion-v-1-3 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 195,000 steps at resolution 512x512 on \"laion-improved-aesthetics\" and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling."
    )
    StableDiffusion1_4 = DiffusionModelInfo(
        "sd-v1-4", 
        "Stable Diffusion v1.4", 
        "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\n The Stable-Diffusion-v-1-4 checkpoint was initialized with the weights of the Stable-Diffusion-v-1-2 checkpoint and subsequently fine-tuned on 225k steps at resolution 512x512 on \"laion-aesthetics v2 5+\" and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling."
    )
    StableDiffusion1_5 = DiffusionModelInfo(
        "v1-5-pruned-emaonly", 
        "Stable Diffusion v1.5", 
        "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\n The Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on \"laion-aesthetics v2 5+\" and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling."
    )
    WaifuDiffusion1_3 = DiffusionModelInfo(
        "wd-v1-3-float32",
        "Waifu Diffusion v1.3",
        "https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1",
        "https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-float32.ckpt",
        "Waifu Diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.\nThe model originally used for fine-tuning is Stable Diffusion 1.4, which is a latent image diffusion model trained on LAION2B-en. The current model has been fine-tuned with a learning rate of 5.0e-6 for 10 epochs on 680k anime-styled images."
    )
    NovelAI_Full = DiffusionModelInfo(
        "nai-full-pruned",
        "Novel AI Full leaked",
        "https://blog.novelai.net/image-generation-announcement-807b3cf0afec",
        "magnet:?xt=urn:btih:5bde442da86265b670a3e5ea3163afad2c6f8ecc&dn=novelaileak",
        "NovelAI Image Generation is a proprietary model that was leaked."
    )
    HentaiAI = DiffusionModelInfo(
        "HD-16",
        "Hentai Diffusion v16",
        "https://github.com/Delcos/Hentai-Diffusion",
        "https://huggingface.co/Deltaadams/Hentai-Diffusion/resolve/main/HD-16.ckpt",
        "Hentai Diffusion has been made to focus not only on hentai, but better hands and better obscure poses."
    )
    # TODO: T5Tokenizer isn't quite working, get back to this when I understand this stuff better
    # JapaneseStableDiffusion = DiffusionModelInfo(
    #     "JSD-v1-4",
    #     "sd-base", # is this correct?
    #     "Japanese Stable Diffusion",
    #     "https://huggingface.co/rinna/japanese-stable-diffusion",
    #     "https://huggingface.co/rinna/japanese-stable-diffusion/tree/main",
    #     "Japanese Stable Diffusion is a Japanese-specific latent text-to-image diffusion model capable of generating photo-realistic images given any text input. Trained on approximately 100 million images with Japanese captions, including the Japanese subset of LAION-5B."
    # )

class DiffusionModel(StableDiffusion):
    def __init__(self, model: DIFF_MODEL):
        # Setup model according to config
        super().__init__()
        
        # Turn off training, switch to evaluation mode
        self.eval()
