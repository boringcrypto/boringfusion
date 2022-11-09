from core.data import ModelMap, ModelLayersInfo

map = ModelMap({
    "5770c902": ModelLayersInfo(
        "CLIP Encoder",
        "sd-wikiart-v2",
        "data/clip-text-encoder/5770c902-5770c902.bin",
        "5770c902",
        "5770c902",
        "torch.float16, torch.int64",
        "123060557",
        "",
        "",
        "",
        ['sd-wikiart-v2', 'bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt', 'bondage3_24000.ckpt', 'easter_e5.ckpt', 'furry_epoch4.ckpt', 'gape22_yiffy15.ckpt', 'HD-16.ckpt', 'last-pruned.ckpt', 'LD-70k-1e-pruned.ckpt', 'Lewd-diffusion-pruned.ckpt', 'pyros-bj-v1-0.ckpt', 'r34_e4.ckpt', 'trinart2_step115000.ckpt', 'trinart2_step60000.ckpt', 'trinart2_step95000.ckpt', 'trinart_characters_it4_v1.ckpt', 'wd-v1-3-float16.ckpt', 'yiffy-e18.ckpt', 'Zack3D_Kinky-v1.ckpt'],
    ),
    "3a64003d": ModelLayersInfo(
        "VAE Encoder",
        "sd-wikiart-v2",
        "data/vae-encoder/3a64003d-3a64003d.bin",
        "3a64003d",
        "3a64003d",
        "torch.float16",
        "34163592",
        "",
        "",
        "",
        ['sd-wikiart-v2'],
    ),
    "4542c082": ModelLayersInfo(
        "VAE Decoder",
        "sd-wikiart-v2",
        "data/vae-decoder/4542c082-4542c082.bin",
        "4542c082",
        "4542c082",
        "torch.float16",
        "49490179",
        "",
        "",
        "",
        ['sd-wikiart-v2'],
    ),
    "72a7f5e3": ModelLayersInfo(
        "UNet",
        "sd-wikiart-v2",
        "data/unet/72a7f5e3-72a7f5e3.bin",
        "72a7f5e3",
        "72a7f5e3",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['sd-wikiart-v2'],
    ),
    "8b408781": ModelLayersInfo(
        "CLIP Encoder",
        "stable-diffusion-v1-4",
        "data/clip-text-encoder/5770c902-8b408781.bin",
        "5770c902",
        "8b408781",
        "torch.float32, torch.int64",
        "123060557",
        "",
        "",
        "",
        ['stable-diffusion-v1-4', 'bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt', 'ema-only-epoch=000142.ckpt', 'gape60.ckpt', 'gg1342_testrun1_pruned.ckpt', 'LOKEAN_MISSIONARY_POV.ckpt', 'LOKEAN_PUPPYSTYLE_POV.ckpt', 'Mixed.ckpt', 'nai-full-latest.ckpt', 'nai-full-pruned.ckpt', 'sd-v1-1-full-ema.ckpt', 'sd-v1-1.ckpt', 'sd-v1-2-full-ema.ckpt', 'sd-v1-2.ckpt', 'sd-v1-3-full-ema.ckpt', 'sd-v1-3.ckpt', 'sd-v1-4-full-ema.ckpt', 'sd-v1-4.ckpt', 'sd-v1-5-inpainting.ckpt', 'v1-5-pruned-emaonly.ckpt', 'v1-5-pruned.ckpt', 'wd-v1-3-float32.ckpt', 'wd-v1-3-full-opt.ckpt', 'wd-v1-3-full.ckpt'],
    ),
    "be3e2ddc": ModelLayersInfo(
        "VAE Encoder",
        "stable-diffusion-v1-4",
        "data/vae-encoder/3a64003d-be3e2ddc.bin",
        "3a64003d",
        "be3e2ddc",
        "torch.float32",
        "34163592",
        "",
        "",
        "",
        ['stable-diffusion-v1-4', 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1'],
    ),
    "fadeb90d": ModelLayersInfo(
        "VAE Decoder",
        "stable-diffusion-v1-4",
        "data/vae-decoder/4542c082-fadeb90d.bin",
        "4542c082",
        "fadeb90d",
        "torch.float32",
        "49490179",
        "",
        "",
        "",
        ['stable-diffusion-v1-4', 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1'],
    ),
    "7da99596": ModelLayersInfo(
        "UNet",
        "Stable Diffusion v1.4 EMA fp32",
        "data/unet/8b21d596-7da99596.bin",
        "8b21d596",
        "7da99596",
        "torch.float32",
        "859520964",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The Stable-Diffusion-v-1-4 checkpoint was initialized with the weights of the Stable-Diffusion-v-1-2 checkpoint and subsequently fine-tuned on 225k steps at resolution 512x512 on 'laion-aesthetics v2 5+' and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling.",
        ['stable-diffusion-v1-4', 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1', 'sd-v1-4-full-ema.ckpt', 'sd-v1-4.ckpt'],
    ),
    "dacf2615": ModelLayersInfo(
        "VAE Encoder",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt",
        "data/vae-encoder/dacf2615-dacf2615.bin",
        "dacf2615",
        "dacf2615",
        "torch.float16",
        "34163664",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt', 'bondage3_24000.ckpt', 'bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt', 'Cyberpunk-Anime-Diffusion.ckpt', 'DCAUV1.ckpt', 'easter_e5.ckpt', 'furry_epoch4.ckpt', 'gape22_yiffy15.ckpt', 'Hiten girl_anime_8k_wallpaper_4k.ckpt', 'last-pruned.ckpt', 'LD-70k-1e-pruned.ckpt', 'Lewd-diffusion-pruned.ckpt', 'pyros-bj-v1-0.ckpt', 'r34_e4.ckpt', 'trinart2_step115000.ckpt', 'trinart2_step60000.ckpt', 'trinart2_step95000.ckpt', 'trinart_characters_it4_v1.ckpt', 'wd-v1-3-float16.ckpt', 'yiffy-e18.ckpt', 'Zack3D_Kinky-v1.ckpt', 'sd-wikiart-v2'],
    ),
    "97dd98b1": ModelLayersInfo(
        "VAE Decoder",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt",
        "data/vae-decoder/97dd98b1-97dd98b1.bin",
        "97dd98b1",
        "97dd98b1",
        "torch.float16",
        "49490199",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt', 'bondage3_24000.ckpt', 'bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt', 'Cyberpunk-Anime-Diffusion.ckpt', 'DCAUV1.ckpt', 'easter_e5.ckpt', 'furry_epoch4.ckpt', 'gape22_yiffy15.ckpt', 'Hiten girl_anime_8k_wallpaper_4k.ckpt', 'last-pruned.ckpt', 'LD-70k-1e-pruned.ckpt', 'Lewd-diffusion-pruned.ckpt', 'pyros-bj-v1-0.ckpt', 'r34_e4.ckpt', 'trinart2_step115000.ckpt', 'trinart2_step60000.ckpt', 'trinart2_step95000.ckpt', 'trinart_characters_it4_v1.ckpt', 'wd-v1-3-float16.ckpt', 'yiffy-e18.ckpt', 'Zack3D_Kinky-v1.ckpt', 'sd-wikiart-v2'],
    ),
    "a6c1993d": ModelLayersInfo(
        "UNet",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt",
        "data/unet/a6c1993d-a6c1993d.bin",
        "a6c1993d",
        "a6c1993d",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt'],
    ),
    "56ea7f90": ModelLayersInfo(
        "VAE Encoder",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt",
        "data/vae-encoder/dacf2615-56ea7f90.bin",
        "dacf2615",
        "56ea7f90",
        "torch.float32",
        "34163664",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt', 'ema-only-epoch=000142.ckpt', 'gape60.ckpt', 'gg1342_testrun1_pruned.ckpt', 'JSD-v1-4.ckpt', 'LOKEAN_MISSIONARY_POV.ckpt', 'LOKEAN_PUPPYSTYLE_POV.ckpt', 'Mixed.ckpt', 'nai-full-latest.ckpt', 'nai-full-pruned.ckpt', 'pachu_artwork_style_v1_iter8000.ckpt', 'robo-diffusion-v1.ckpt', 'sd-v1-1-full-ema.ckpt', 'sd-v1-1.ckpt', 'sd-v1-2-full-ema.ckpt', 'sd-v1-2.ckpt', 'sd-v1-3-full-ema.ckpt', 'sd-v1-3.ckpt', 'sd-v1-4-full-ema.ckpt', 'sd-v1-4.ckpt', 'sd-v1-5-inpainting.ckpt', 'v1-5-pruned-emaonly.ckpt', 'v1-5-pruned.ckpt', 'wd-v1-3-float32.ckpt', 'wd-v1-3-full-opt.ckpt', 'wd-v1-3-full.ckpt', 'stable-diffusion-v1-4', 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1'],
    ),
    "8b7877f3": ModelLayersInfo(
        "VAE Decoder",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt",
        "data/vae-decoder/97dd98b1-8b7877f3.bin",
        "97dd98b1",
        "8b7877f3",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt', 'ema-only-epoch=000142.ckpt', 'gape60.ckpt', 'gg1342_testrun1_pruned.ckpt', 'JSD-v1-4.ckpt', 'LOKEAN_MISSIONARY_POV.ckpt', 'LOKEAN_PUPPYSTYLE_POV.ckpt', 'Mixed.ckpt', 'nai-full-latest.ckpt', 'nai-full-pruned.ckpt', 'robo-diffusion-v1.ckpt', 'sd-v1-1-full-ema.ckpt', 'sd-v1-1.ckpt', 'sd-v1-2-full-ema.ckpt', 'sd-v1-2.ckpt', 'sd-v1-3-full-ema.ckpt', 'sd-v1-3.ckpt', 'sd-v1-4-full-ema.ckpt', 'sd-v1-4.ckpt', 'sd-v1-5-inpainting.ckpt', 'v1-5-pruned-emaonly.ckpt', 'v1-5-pruned.ckpt', 'wd-v1-3-float32.ckpt', 'wd-v1-3-full-opt.ckpt', 'wd-v1-3-full.ckpt', 'stable-diffusion-v1-4', 'Taiyi-Stable-Diffusion-1B-Chinese-v0.1'],
    ),
    "8592715d": ModelLayersInfo(
        "UNet",
        "bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt",
        "data/unet/a6c1993d-8592715d.bin",
        "a6c1993d",
        "8592715d",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['bf_fb_v3_t4_b16_noadd-ema-pruned-fp32.ckpt'],
    ),
    "1fa05e1e": ModelLayersInfo(
        "UNet EMA",
        "bondage3_24000.ckpt",
        "data/unet/1fa05e1e-1fa05e1e.bin",
        "1fa05e1e",
        "1fa05e1e",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['bondage3_24000.ckpt'],
    ),
    "ef4ab8be": ModelLayersInfo(
        "CLIP Encoder",
        "bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt",
        "data/clip-text-encoder/ef4ab8be-ef4ab8be.bin",
        "ef4ab8be",
        "ef4ab8be",
        "torch.float16",
        "123060557",
        "",
        "",
        "",
        ['bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt'],
    ),
    "1223ec31": ModelLayersInfo(
        "UNet",
        "bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt",
        "data/unet/1223ec31-1223ec31.bin",
        "1223ec31",
        "1223ec31",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['bukkake_20_training_images_2020_max_training_steps_woman_class_word.ckpt'],
    ),
    "0ec000aa": ModelLayersInfo(
        "CLIP Encoder",
        "cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt",
        "data/clip-text-encoder/0ec000aa-0ec000aa.bin",
        "0ec000aa",
        "0ec000aa",
        "torch.float32, torch.int64",
        "123060557",
        "",
        "",
        "",
        ['cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt'],
    ),
    "67084cde": ModelLayersInfo(
        "VAE Encoder",
        "cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt",
        "data/vae-encoder/67084cde-67084cde.bin",
        "67084cde",
        "67084cde",
        "torch.float32",
        "34163664",
        "",
        "",
        "",
        ['cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt'],
    ),
    "fab450bf": ModelLayersInfo(
        "VAE Decoder",
        "cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt",
        "data/vae-decoder/fab450bf-fab450bf.bin",
        "fab450bf",
        "fab450bf",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt'],
    ),
    "c766bbed": ModelLayersInfo(
        "UNet",
        "cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt",
        "data/unet/c766bbed-c766bbed.bin",
        "c766bbed",
        "c766bbed",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['cookie_sd_pony_run_a12_datasetv5_300k_imgs_fp32.ckpt'],
    ),
    "e05242e9": ModelLayersInfo(
        "CLIP Encoder",
        "Cyberpunk-Anime-Diffusion.ckpt",
        "data/clip-text-encoder/e05242e9-e05242e9.bin",
        "e05242e9",
        "e05242e9",
        "torch.float16",
        "123060557",
        "",
        "",
        "",
        ['Cyberpunk-Anime-Diffusion.ckpt'],
    ),
    "796e54bf": ModelLayersInfo(
        "UNet",
        "Cyberpunk-Anime-Diffusion.ckpt",
        "data/unet/796e54bf-796e54bf.bin",
        "796e54bf",
        "796e54bf",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['Cyberpunk-Anime-Diffusion.ckpt'],
    ),
    "116a6e0a": ModelLayersInfo(
        "CLIP Encoder",
        "DCAUV1.ckpt",
        "data/clip-text-encoder/116a6e0a-116a6e0a.bin",
        "116a6e0a",
        "116a6e0a",
        "torch.float16",
        "123060557",
        "",
        "",
        "",
        ['DCAUV1.ckpt'],
    ),
    "7acebf57": ModelLayersInfo(
        "UNet",
        "DCAUV1.ckpt",
        "data/unet/7acebf57-7acebf57.bin",
        "7acebf57",
        "7acebf57",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['DCAUV1.ckpt'],
    ),
    "6f403e4a": ModelLayersInfo(
        "UNet",
        "easter_e5.ckpt",
        "data/unet/6f403e4a-6f403e4a.bin",
        "6f403e4a",
        "6f403e4a",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['easter_e5.ckpt'],
    ),
    "55a0370c": ModelLayersInfo(
        "UNet",
        "ema-only-epoch=000142.ckpt",
        "data/unet/0cd99ff3-55a0370c.bin",
        "0cd99ff3",
        "55a0370c",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['ema-only-epoch=000142.ckpt'],
    ),
    "89ab3fc3": ModelLayersInfo(
        "CLIP Encoder",
        "f111.ckpt",
        "data/clip-text-encoder/5770c902-89ab3fc3.bin",
        "5770c902",
        "89ab3fc3",
        "torch.float32",
        "123060557",
        "",
        "",
        "",
        ['f111.ckpt'],
    ),
    "b2c3db47": ModelLayersInfo(
        "VAE Encoder",
        "f111.ckpt",
        "data/vae-encoder/8159f4e7-b2c3db47.bin",
        "8159f4e7",
        "b2c3db47",
        "torch.float32",
        "34163664",
        "",
        "",
        "",
        ['f111.ckpt'],
    ),
    "bdd84115": ModelLayersInfo(
        "VAE Decoder",
        "f111.ckpt",
        "data/vae-decoder/97dd98b1-bdd84115.bin",
        "97dd98b1",
        "bdd84115",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['f111.ckpt'],
    ),
    "9b3ee072": ModelLayersInfo(
        "UNet",
        "f111.ckpt",
        "data/unet/23322baa-9b3ee072.bin",
        "23322baa",
        "9b3ee072",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['f111.ckpt'],
    ),
    "d996997f": ModelLayersInfo(
        "CLIP Encoder",
        "f222.ckpt",
        "data/clip-text-encoder/5770c902-d996997f.bin",
        "5770c902",
        "d996997f",
        "torch.float32",
        "123060557",
        "",
        "",
        "",
        ['f222.ckpt'],
    ),
    "e642e16e": ModelLayersInfo(
        "VAE Encoder",
        "f222.ckpt",
        "data/vae-encoder/8159f4e7-e642e16e.bin",
        "8159f4e7",
        "e642e16e",
        "torch.float32",
        "34163664",
        "",
        "",
        "",
        ['f222.ckpt'],
    ),
    "cbc41d9b": ModelLayersInfo(
        "VAE Decoder",
        "f222.ckpt",
        "data/vae-decoder/97dd98b1-cbc41d9b.bin",
        "97dd98b1",
        "cbc41d9b",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['f222.ckpt'],
    ),
    "5a623fe1": ModelLayersInfo(
        "UNet",
        "f222.ckpt",
        "data/unet/134a683d-5a623fe1.bin",
        "134a683d",
        "5a623fe1",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['f222.ckpt'],
    ),
    "06d65eaf": ModelLayersInfo(
        "UNet EMA",
        "furry_epoch4.ckpt",
        "data/unet/06d65eaf-06d65eaf.bin",
        "06d65eaf",
        "06d65eaf",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['furry_epoch4.ckpt'],
    ),
    "9355b422": ModelLayersInfo(
        "UNet EMA",
        "gape22_yiffy15.ckpt",
        "data/unet/9355b422-9355b422.bin",
        "9355b422",
        "9355b422",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['gape22_yiffy15.ckpt'],
    ),
    "3dde4f97": ModelLayersInfo(
        "UNet",
        "gape60.ckpt",
        "data/unet/9d0e893c-3dde4f97.bin",
        "9d0e893c",
        "3dde4f97",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['gape60.ckpt'],
    ),
    "61a460c5": ModelLayersInfo(
        "UNet",
        "gg1342_testrun1_pruned.ckpt",
        "data/unet/3b67e187-61a460c5.bin",
        "3b67e187",
        "61a460c5",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['gg1342_testrun1_pruned.ckpt'],
    ),
    "191d2210": ModelLayersInfo(
        "VAE Encoder",
        "HD-16.ckpt",
        "data/vae-encoder/10acfaa6-191d2210.bin",
        "10acfaa6",
        "191d2210",
        "torch.float32",
        "34163664",
        "",
        "",
        "",
        ['HD-16.ckpt'],
    ),
    "05e8c698": ModelLayersInfo(
        "VAE Decoder",
        "HD-16.ckpt",
        "data/vae-decoder/36abe8b0-05e8c698.bin",
        "36abe8b0",
        "05e8c698",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['HD-16.ckpt'],
    ),
    "5528a400": ModelLayersInfo(
        "UNet EMA",
        "Hentai Diffusion v16",
        "data/unet/5528a400-5528a400.bin",
        "5528a400",
        "5528a400",
        "torch.float16",
        "859520964",
        "https://github.com/Delcos/Hentai-Diffusion",
        "https://huggingface.co/Deltaadams/Hentai-Diffusion/resolve/main/HD-16.ckpt",
        "Hentai Diffusion has been made to focus not only on hentai, but better hands and better obscure poses.",
        ['HD-16.ckpt', 'Zack3D_Kinky-v1.ckpt'],
    ),
    "009519cc": ModelLayersInfo(
        "CLIP Encoder",
        "Hiten girl_anime_8k_wallpaper_4k.ckpt",
        "data/clip-text-encoder/009519cc-009519cc.bin",
        "009519cc",
        "009519cc",
        "torch.float16",
        "123060557",
        "",
        "",
        "",
        ['Hiten girl_anime_8k_wallpaper_4k.ckpt'],
    ),
    "325017eb": ModelLayersInfo(
        "UNet",
        "Hiten girl_anime_8k_wallpaper_4k.ckpt",
        "data/unet/325017eb-325017eb.bin",
        "325017eb",
        "325017eb",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['Hiten girl_anime_8k_wallpaper_4k.ckpt'],
    ),
    "e5611d52": ModelLayersInfo(
        "CLIP Encoder",
        "JSD-v1-4.ckpt",
        "data/clip-text-encoder/69cfcfc3-e5611d52.bin",
        "69cfcfc3",
        "e5611d52",
        "torch.float32, torch.int64",
        "109691213",
        "",
        "",
        "",
        ['JSD-v1-4.ckpt'],
    ),
    "ab4e402c": ModelLayersInfo(
        "UNet",
        "Japanese Stable Diffusion 1.4",
        "data/unet/cad4183c-ab4e402c.bin",
        "cad4183c",
        "ab4e402c",
        "torch.float32",
        "859520964",
        "https://huggingface.co/rinna/japanese-stable-diffusion",
        "https://huggingface.co/rinna/japanese-stable-diffusion/tree/main",
        "Japanese Stable Diffusion is a Japanese-specific latent text-to-image diffusion model capable of generating photo-realistic images given any text input. Trained on approximately 100 million images with Japanese captions, including the Japanese subset of LAION-5B.",
        ['JSD-v1-4.ckpt'],
    ),
    "4e03132f": ModelLayersInfo(
        "UNet EMA",
        "last-pruned.ckpt",
        "data/unet/4e03132f-4e03132f.bin",
        "4e03132f",
        "4e03132f",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['last-pruned.ckpt'],
    ),
    "2ee3899d": ModelLayersInfo(
        "UNet EMA",
        "LD-70k-1e-pruned.ckpt",
        "data/unet/2ee3899d-2ee3899d.bin",
        "2ee3899d",
        "2ee3899d",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['LD-70k-1e-pruned.ckpt'],
    ),
    "ab8f2115": ModelLayersInfo(
        "UNet EMA",
        "Lewd-diffusion-pruned.ckpt",
        "data/unet/ab8f2115-ab8f2115.bin",
        "ab8f2115",
        "ab8f2115",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['Lewd-diffusion-pruned.ckpt'],
    ),
    "d81fcda3": ModelLayersInfo(
        "UNet",
        "LOKEAN_MISSIONARY_POV.ckpt",
        "data/unet/ebfad7a8-d81fcda3.bin",
        "ebfad7a8",
        "d81fcda3",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['LOKEAN_MISSIONARY_POV.ckpt'],
    ),
    "15e554a5": ModelLayersInfo(
        "UNet",
        "LOKEAN_PUPPYSTYLE_POV.ckpt",
        "data/unet/721f9e07-15e554a5.bin",
        "721f9e07",
        "15e554a5",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['LOKEAN_PUPPYSTYLE_POV.ckpt'],
    ),
    "130d74ce": ModelLayersInfo(
        "UNet",
        "Mixed.ckpt",
        "data/unet/ab4ac737-130d74ce.bin",
        "ab4ac737",
        "130d74ce",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['Mixed.ckpt'],
    ),
    "cedfeefb": ModelLayersInfo(
        "UNet EMA",
        "Novel AI Full leaked",
        "data/unet/ef56f0fa-cedfeefb.bin",
        "ef56f0fa",
        "cedfeefb",
        "torch.float32",
        "859520964",
        "https://blog.novelai.net/image-generation-announcement-807b3cf0afec",
        "magnet:?xt=urn:btih:5bde442da86265b670a3e5ea3163afad2c6f8ecc&dn=novelaileak",
        "NovelAI Image Generation is a proprietary model that was leaked.",
        ['nai-full-latest.ckpt', 'nai-full-pruned.ckpt'],
    ),
    "b44c8b0e": ModelLayersInfo(
        "CLIP Encoder",
        "pachu_artwork_style_v1_iter8000.ckpt",
        "data/clip-text-encoder/13835133-b44c8b0e.bin",
        "13835133",
        "b44c8b0e",
        "torch.float32, torch.int64",
        "123060557",
        "",
        "",
        "",
        ['pachu_artwork_style_v1_iter8000.ckpt'],
    ),
    "62ab9f03": ModelLayersInfo(
        "VAE Decoder",
        "pachu_artwork_style_v1_iter8000.ckpt",
        "data/vae-decoder/75335bbc-62ab9f03.bin",
        "75335bbc",
        "62ab9f03",
        "torch.float32",
        "49490199",
        "",
        "",
        "",
        ['pachu_artwork_style_v1_iter8000.ckpt'],
    ),
    "b103a9fc": ModelLayersInfo(
        "UNet",
        "pachu_artwork_style_v1_iter8000.ckpt",
        "data/unet/6706c7a0-b103a9fc.bin",
        "6706c7a0",
        "b103a9fc",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['pachu_artwork_style_v1_iter8000.ckpt'],
    ),
    "b8a1e44c": ModelLayersInfo(
        "UNet",
        "pyros-bj-v1-0.ckpt",
        "data/unet/b8a1e44c-b8a1e44c.bin",
        "b8a1e44c",
        "b8a1e44c",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['pyros-bj-v1-0.ckpt'],
    ),
    "6910d82d": ModelLayersInfo(
        "UNet",
        "r34_e4.ckpt",
        "data/unet/6910d82d-6910d82d.bin",
        "6910d82d",
        "6910d82d",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['r34_e4.ckpt'],
    ),
    "72e47874": ModelLayersInfo(
        "CLIP Encoder",
        "robo-diffusion-v1.ckpt",
        "data/clip-text-encoder/ed8e40b3-72e47874.bin",
        "ed8e40b3",
        "72e47874",
        "torch.float32, torch.int64",
        "123060557",
        "",
        "",
        "",
        ['robo-diffusion-v1.ckpt'],
    ),
    "cf749104": ModelLayersInfo(
        "UNet",
        "robo-diffusion-v1.ckpt",
        "data/unet/3a6bfe5c-cf749104.bin",
        "3a6bfe5c",
        "cf749104",
        "torch.float32",
        "859520964",
        "",
        "",
        "",
        ['robo-diffusion-v1.ckpt'],
    ),
    "062d85c8": ModelLayersInfo(
        "UNet EMA",
        "Stable Diffusion v1.1 EMA fp32",
        "data/unet/7d1d7e93-062d85c8.bin",
        "7d1d7e93",
        "062d85c8",
        "torch.float32",
        "859520964",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-1-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The Stable-Diffusion-v-1-1 was trained on 237,000 steps at resolution 256x256 on laion2B-en, followed by 194,000 steps at resolution 512x512 on laion-high-resolution (170M examples from LAION-5B with resolution >= 1024x1024).",
        ['sd-v1-1-full-ema.ckpt', 'sd-v1-1.ckpt'],
    ),
    "161f6a95": ModelLayersInfo(
        "UNet EMA",
        "Stable Diffusion v1.2 EMA fp32",
        "data/unet/19de9b69-161f6a95.bin",
        "19de9b69",
        "161f6a95",
        "torch.float32",
        "859520964",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-2-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The Stable-Diffusion-v-1-2 checkpoint was initialized with the weights of the Stable-Diffusion-v-1-1 checkpoint and subsequently fine-tuned on 515,000 steps at resolution 512x512 on 'laion-improved-aesthetics' (a subset of laion2B-en, filtered to images with an original size >= 512x512, estimated aesthetics score > 5.0, and an estimated watermark probability < 0.5.",
        ['sd-v1-2-full-ema.ckpt', 'sd-v1-2.ckpt'],
    ),
    "354e3712": ModelLayersInfo(
        "UNet EMA",
        "Stable Diffusion v1.3 EMA fp32",
        "data/unet/d4e70cc4-354e3712.bin",
        "d4e70cc4",
        "354e3712",
        "torch.float32",
        "859520964",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-3-original",
        "https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The Stable-Diffusion-v-1-3 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 195,000 steps at resolution 512x512 on 'laion-improved-aesthetics' and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling.",
        ['sd-v1-3-full-ema.ckpt', 'sd-v1-3.ckpt'],
    ),
    "eaa3719c": ModelLayersInfo(
        "UNet",
        "sd-v1-5-inpainting.ckpt",
        "data/unet/6e119516-eaa3719c.bin",
        "6e119516",
        "eaa3719c",
        "torch.float32",
        "859535364",
        "",
        "",
        "",
        ['sd-v1-5-inpainting.ckpt'],
    ),
    "0c71c262": ModelLayersInfo(
        "UNet",
        "trinart2_step115000.ckpt",
        "data/unet/0c71c262-0c71c262.bin",
        "0c71c262",
        "0c71c262",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['trinart2_step115000.ckpt'],
    ),
    "782de700": ModelLayersInfo(
        "UNet",
        "trinart2_step60000.ckpt",
        "data/unet/782de700-782de700.bin",
        "782de700",
        "782de700",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['trinart2_step60000.ckpt'],
    ),
    "976c48c9": ModelLayersInfo(
        "UNet",
        "trinart2_step95000.ckpt",
        "data/unet/976c48c9-976c48c9.bin",
        "976c48c9",
        "976c48c9",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['trinart2_step95000.ckpt'],
    ),
    "fb4f2006": ModelLayersInfo(
        "UNet",
        "trinart_characters_it4_v1.ckpt",
        "data/unet/fb4f2006-fb4f2006.bin",
        "fb4f2006",
        "fb4f2006",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['trinart_characters_it4_v1.ckpt'],
    ),
    "8d39156d": ModelLayersInfo(
        "UNet",
        "Stable Diffusion v1.5 EMA fp32",
        "data/unet/1fa05e1e-8d39156d.bin",
        "1fa05e1e",
        "8d39156d",
        "torch.float32",
        "859520964",
        "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
        "Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on 'laion-aesthetics v2 5+' and 10\% dropping of the text-conditioning to improve classifier-free guidance sampling.",
        ['v1-5-pruned-emaonly.ckpt', 'v1-5-pruned.ckpt'],
    ),
    "fbd17571": ModelLayersInfo(
        "UNet",
        "Waifu Diffusion v1.3 EMA fp16",
        "data/unet/fbd17571-fbd17571.bin",
        "fbd17571",
        "fbd17571",
        "torch.float16",
        "859520964",
        "https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1",
        "https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-float16.ckpt",
        "Waifu Diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning. The model originally used for fine-tuning is Stable Diffusion 1.4, which is a latent image diffusion model trained on LAION2B-en. The current model has been fine-tuned with a learning rate of 5.0e-6 for 10 epochs on 680k anime-styled images.",
        ['wd-v1-3-float16.ckpt'],
    ),
    "6a9f373b": ModelLayersInfo(
        "UNet",
        "Waifu Diffusion v1.3 EMA fp32",
        "data/unet/fbd17571-6a9f373b.bin",
        "fbd17571",
        "6a9f373b",
        "torch.float32",
        "859520964",
        "https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1",
        "https://huggingface.co/hakurei/waifu-diffusion-v1-3/blob/main/wd-v1-3-float32.ckpt",
        "Waifu Diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning. The model originally used for fine-tuning is Stable Diffusion 1.4, which is a latent image diffusion model trained on LAION2B-en. The current model has been fine-tuned with a learning rate of 5.0e-6 for 10 epochs on 680k anime-styled images.",
        ['wd-v1-3-float32.ckpt', 'wd-v1-3-full-opt.ckpt', 'wd-v1-3-full.ckpt'],
    ),
    "09da1114": ModelLayersInfo(
        "UNet EMA",
        "yiffy-e18.ckpt",
        "data/unet/09da1114-09da1114.bin",
        "09da1114",
        "09da1114",
        "torch.float16",
        "859520964",
        "",
        "",
        "",
        ['yiffy-e18.ckpt'],
    )
})