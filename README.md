# boringfusion
Stable Diffusion Library

As a learning exercise I'm refactoring the original stable diffusion repo to be a lot simpler and remove anything not absolutely needed. I'm still learning about neural networks, diffusion, pytorxh, cuda, etc.

Hope this is useful to others trying to understand stable diffusion. And maybe it will grow into something useful for coders wishing to use stable diffusion in their code.


# Thanks
Used code/learnings from:
- https://github.com/CompVis/stable-diffusion
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/openai/improved-diffusion
- https://github.com/CompVis/taming-transformers
- https://github.com/openai/guided-diffusion

Thanks for your help and insights:
CodeExplode Nerfy#5590 - Scigyric#8427

# Cool resources
- https://upscale.wiki/wiki/Model_Database
- https://rentry.co/sdmodels
- https://rentry.co/sdupdates
- https://rentry.org/lftbl

# Learnings

## Dictionary
Tensor
: Basically a multi-dimensional (1 to many) array of numbers.
Embedding
: Just a fancy word for converting into some numerical representation.

## Inference

### Step 1. CLIP Embedding
The first step in inference is to convert the text prompt and negative text prompt to CLIP embeddings.

CLIP is a model created by OpenAI that is trained on 400M+ images and text prompts. The model consists of 2 parts:
- a text encoder that will embed text into an embedding (Tensor of size [1, 77, 768]).
- an image encoder that will embed images into an embedding (Tensor of size [1, 77, 768]).

The embeddings are a directional vector defining a concept, in 768-dimensional space, and if you scale them up or down the concept gets stronger or weaker. SD takes 77 concepts, starting with the "start token" and ending with the "end token". Unused spots are also filled with the "end token". So there are 75 usable tokens.

In SD the CLIP text embedder is used, it has 2 steps:
- The CLIPTokenizer turns the text prompt into 77 tokens. There are 49407 distinct tokens. These are integer numbers that can be looked up here:
  https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json or by calling `.get_vocab()` on the `CLIPTokenizer`.
- Then the transformer turns these 77 tokens into vectors in 768-dimensional space.

### Example

Text: `"photo of a couch, photorealistic"`

Tokens: `tensor([[49406, 1125, 539, 320, 12724, 267, 1153, 16157, 49407, 49407, 49407, ..., 49407]], device='cuda:0')`

Lookup up in vocab, this is: `['<|startoftext|>', 'photo</w>', 'of</w>', 'a</w>', 'couch</w>', ',</w>', 'photo', 'realistic</w>', '<|endoftext|>', '<|endoftext|>', ..., '<|endoftext|>']`

### Further Reading
https://openai.com/blog/clip/
: The original blog post
https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html : a clearer explanation




