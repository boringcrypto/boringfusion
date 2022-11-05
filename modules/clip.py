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

        self.token_embedding = self.transformer.get_input_embeddings()

        position_ids = self.transformer.text_model.embeddings.position_ids[:, :77]
        self.position_embeddings = self.transformer.text_model.embeddings.position_embedding(position_ids)        

        self.vocab = {value:key for (key,value) in self.tokenizer.get_vocab().items()}

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

class EmbeddingBuilder():
    def __init__(self, clip_embedder: CLIPEmbedder) -> None:
        self.clip_embedder = clip_embedder

        tokenizer_output = clip_embedder.tokenizer(
            "", # the prompt or array of prompts to tokenize
            truncation=True, # truncate if longer than 77 tokens
            max_length=77, # Stable Diffusion uses 77 tokens
            padding="max_length", # If there are less than 77 tokens, pad to 77 tokens
            return_tensors="pt" # Return pytorch tensor
        )

        self.token_embeddings = self.clip_embedder.token_embedding(tokenizer_output.input_ids)
        self.position = 1

    def _embed(self, prompt):
        tokenizer_output = self.clip_embedder.tokenizer(
            prompt, # the prompt or array of prompts to tokenize
            truncation=True, # truncate if longer than 77 tokens
            max_length=76 - self.position, # Stable Diffusion uses 77 tokens
            padding="do_not_pad", # No padding
            return_tensors="pt" # Return pytorch tensor
        )
        return self.clip_embedder.token_embedding(tokenizer_output.input_ids[:, 1:-1])

    def _append_embeddings(self, embeddings):
        self.token_embeddings[0, self.position:self.position+embeddings.size(1), :] = embeddings
        self.position += embeddings.size(1)

    def add_prompt(self, prompt, weight=1.0):
        embeddings = self._embed(prompt) * weight
        self._append_embeddings(embeddings)

    def add_combined_prompt(self, prompts, weights, weight=1.0):
        combined = torch.zeros(1, 1, 768)
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
        input_embeddings = self.token_embeddings + self.clip_embedder.position_embeddings

        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = self.clip_embedder.transformer.text_model._build_causal_attention_mask(bsz, seq_len)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True 
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.clip_embedder.transformer.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.clip_embedder.device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.clip_embedder.transformer.text_model.final_layer_norm(output)

        # And now they're ready!
        return output

    
