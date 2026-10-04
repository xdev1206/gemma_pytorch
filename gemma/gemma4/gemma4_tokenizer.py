# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tokenizer adapter for Gemma 4's Hugging Face tokenizer.json."""

import os
import json
from pathlib import Path
from typing import List


class Gemma4Tokenizer:
    def __init__(self, model_path: str):
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:
            raise ImportError(
                "Gemma 4 tokenization requires the tokenizers package."
            ) from exc
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        self.tokenizer = HFTokenizer.from_file(model_path)
        config_path = os.path.join(os.path.dirname(model_path), "tokenizer_config.json")
        tokenizer_config = {}
        if os.path.isfile(config_path):
            tokenizer_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.n_words = self.tokenizer.get_vocab_size()
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.boi_id = self.tokenizer.token_to_id(
            tokenizer_config.get("boi_token", "<|image>")
        )
        self.eoi_id = self.tokenizer.token_to_id(
            tokenizer_config.get("eoi_token", "<image|>")
        )
        self.boa_id = self.tokenizer.token_to_id(
            tokenizer_config.get("boa_token", "<|audio>")
        )
        self.eoa_id = self.tokenizer.token_to_id(
            tokenizer_config.get("eoa_token", "<audio|>")
        )

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        token_ids = self.tokenizer.encode(text).ids
        if bos:
            token_ids.insert(0, self.bos_id)
        if eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
