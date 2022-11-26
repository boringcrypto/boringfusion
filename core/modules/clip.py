from typing import List
import os
import torch
import torch.nn as nn
import numpy as np
from enum import Enum

from transformers import CLIPTokenizer, CLIPTextModel

from .util import BoringModule, should_run_on_gpu
from open_clip.model import _build_text_tower

TRANSFORMER_STATE_DICT_PATH = "core/data/clip-transformer/pytorch_model.bin"

class CLIPBase(BoringModule):
    def embed(self, prompt, max_length=77, padding="max_length"):
        pass

    def encode(self, input_embeddings):
        pass

    def forward(self, prompt_or_prompts: str or List[str]):
        pass

class OpenCLIP(BoringModule):
    def __init__(self, layers=None):
        """The CLIPEmbedder turns prompts into 77 tokens with 768 dimensions for use with Stable Diffusion.
        By default the model will live in CPU memory and will be moved onto the GPU only while needed. After you
        call `.cuda()` it will live in GPU memory until manually moved back.
        """
        super().__init__()

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers)

        text = _build_text_tower(1024, 
            {
                "context_length": 77,
                "vocab_size": 49408,
                "width": 1024,
                "heads": 16,
                "layers": 24
            }, False, torch.float32)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x


class CLIPEmbedder(BoringModule):
    def __init__(self, layers=None):
        """The CLIPEmbedder turns prompts into 77 tokens with 768 dimensions for use with Stable Diffusion.
        By default the model will live in CPU memory and will be moved onto the GPU only while needed. After you
        call `.cuda()` it will live in GPU memory until manually moved back.
        """
        super().__init__()

        if isinstance(layers, Enum):
            from ..data.model_data import ModelData
            layers = ModelData.load(layers)

        # The tokenizer has a vocabulary of 49407 tokens and runs on the cpu (not a module). Small memory footprint (few MB).
        self.tokenizer = CLIPTokenizer.from_pretrained("core/data/clip-tokenizer", local_files_only=True)
        # The transformer is a module of about 480MB.
        download = layers is None and not os.path.exists(TRANSFORMER_STATE_DICT_PATH)

        if download:
            print("Downloading CLIP Embedder from HuggingFace")
        self.transformer = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14" if download else "core/data/clip-transformer", 
                state_dict=layers, 
                local_files_only=not download
            ).eval()
        if download:
            self.save_default_transformer()            
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

        self.token_embedding = self.transformer.get_input_embeddings()

        position_ids = self.transformer.text_model.embeddings.position_ids[:, :77]
        self.position_embeddings = self.transformer.text_model.embeddings.position_embedding(position_ids)        

        self.vocab = {value:key for (key,value) in self.tokenizer.get_vocab().items()}

    def save_default_transformer(self):
        torch.save(self.transformer.state_dict(), TRANSFORMER_STATE_DICT_PATH)

    @torch.no_grad()
    def embed(self, prompt, max_length=77, padding="max_length"):
        tokenizer_output = self.tokenizer(
            prompt, # the prompt or array of prompts to tokenize
            truncation=True, # truncate if longer than 77 tokens
            max_length=max_length, # Stable Diffusion uses 77 tokens
            padding=padding, # No padding
            return_tensors="pt" # Return pytorch tensor
        )
        input_ids = tokenizer_output.input_ids.to(device=self.device)
        return self.token_embedding(input_ids)

    @should_run_on_gpu
    @torch.no_grad()
    def encode(self, input_embeddings):
        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True 
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.transformer.text_model.encoder(
            inputs_embeds=input_embeddings.to(self.device),
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.transformer.text_model.final_layer_norm(output)

        # And now they're ready!
        return output

    @should_run_on_gpu
    @torch.no_grad()
    def forward(self, prompt_or_prompts: str or List[str]):
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

class PromptBuilder():
    def __init__(self, clip_embedder: CLIPEmbedder) -> None:
        self.clip_embedder = clip_embedder

        self.token_embeddings = self.clip_embedder.embed("")
        self.position = 1

    def _embed(self, prompt):
        return self.clip_embedder.embed(prompt, 76 - self.position, "do_not_pad")[:, 1:-1]

    def _append_embeddings(self, embeddings):
        self.token_embeddings[0, self.position:self.position+embeddings.size(1), :] = embeddings
        self.position += embeddings.size(1)

    def add_prompt(self, prompt, weight=1.0):
        embeddings = self._embed(prompt) * weight
        self._append_embeddings(embeddings)

    def add_combined_prompt(self, prompts, weights, weight=1.0):
        combined = torch.zeros(1, 1, 768, device=self.device)
        total_weights = sum(weights)
        for i, prompt in enumerate(prompts):
            embeddings = self._embed(prompt)
            if combined.size(1) < embeddings.size(1):
                new = torch.zeros(embeddings.shape)
                new[0:, 0:combined.size(1), 0:] = combined
                combined = new
            combined.add_(embeddings * (weights[i] / total_weights))

        self._append_embeddings(combined * weight)

    @property
    def embedding(self):
        input_embeddings \
            = self.token_embeddings.to(device=self.device) \
            + self.clip_embedder.position_embeddings.to(device=self.device)

        return self.clip_embedder.encode(input_embeddings)

    @property
    def device(self):
        return self.clip_embedder.device

    
