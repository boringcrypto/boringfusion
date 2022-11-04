from typing import List
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

from .util import BoringModule, should_run_on_gpu

class CLIPEmbedder(BoringModule):
    def __init__(self):
        """The CLIPEmbedder turns prompts into 77 tokens with 768 dimensions for use with Stable Diffusion.
        By default the model will live in CPU memory and will be moved onto the GPU only while needed. After you
        call `.cuda()` it will live in GPU memory until manually moved back.
        """
        super().__init__()
        # The tokenizer has a vocabulary of 49407 tokens and runs on the cpu (not a module). Small memory footprint (few MB).
        self.tokenizer = CLIPTokenizer.from_pretrained("data/clip-tokenizer", local_files_only=True)
        # The transformer is a module of about 480MB.
        self.transformer = CLIPTextModel.from_pretrained("data/clip-transformer", local_files_only=True).eval()
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

        self.vocab = {value:key for (key,value) in self.tokenizer.get_vocab().items()}

    @should_run_on_gpu
    @torch.no_grad()
    def forward(self, prompt_or_prompts: str or List[str]):
        # Docs: https://huggingface.co/transformers/v4.8.0/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
        tokenizer_output = self.tokenizer(
            prompt_or_prompts, # the prompt or array of prompts to tokenize
            truncation=True, # truncate if longer than 77 tokens
            return_attention_mask=False, # We're not using the attention mask
            max_length=77, # Stable Diffusion uses 77 tokens
            padding="max_length", # If there are less than 77 tokens, pad to 77 tokens
            return_tensors="pt" # Return pytorch tensor
        )
        tokens = tokenizer_output["input_ids"].to(self.transformer.device)
        outputs = self.transformer(input_ids=tokens)

        return outputs.last_hidden_state
