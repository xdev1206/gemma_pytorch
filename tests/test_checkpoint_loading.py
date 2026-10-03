import unittest
from unittest import mock

import torch
from torch import nn

from gemma.checkpoint import load_gemma3_text_safetensors


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(1, 6, bias=False)
        self.o_proj = nn.Linear(1, 1, bias=False)
        self.query_norm = nn.LayerNorm(1, elementwise_affine=True, bias=False)
        self.key_norm = nn.LayerNorm(1, elementwise_affine=True, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttention()
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(1, 1, bias=False),
            "up_proj": nn.Linear(1, 1, bias=False),
            "down_proj": nn.Linear(1, 1, bias=False),
        })
        for name in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        ):
            setattr(self, name, nn.LayerNorm(1, elementwise_affine=True, bias=False))


class _TinyGemma3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(2, 1)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(1, elementwise_affine=True, bias=False)
        self.model.layers = nn.ModuleList([_TinyLayer()])


class _Reader:
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return self.values.keys()

    def get_tensor(self, name):
        return self.values[name]


class CheckpointLoadingTest(unittest.TestCase):
    def test_gemma3_qkv_projection_is_loaded_and_mapped(self):
        values = {
            "model.embed_tokens.weight": torch.ones(2, 1),
            "model.norm.weight": torch.ones(1),
            "model.layers.0.self_attn.q_proj.weight": torch.full((2, 1), 1.0),
            "model.layers.0.self_attn.k_proj.weight": torch.full((2, 1), 2.0),
            "model.layers.0.self_attn.v_proj.weight": torch.full((2, 1), 3.0),
            "model.layers.0.self_attn.o_proj.weight": torch.ones(1, 1),
            "model.layers.0.self_attn.q_norm.weight": torch.ones(1),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(1),
        }
        for name in ("gate_proj", "up_proj", "down_proj"):
            values[f"model.layers.0.mlp.{name}.weight"] = torch.ones(1, 1)
        for name in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        ):
            values[f"model.layers.0.{name}.weight"] = torch.ones(1)

        model = _TinyGemma3()
        with mock.patch("gemma.checkpoint.os.path.isfile", return_value=True), \
                mock.patch("gemma.checkpoint._open_safetensors", return_value=_Reader(values)):
            load_gemma3_text_safetensors(model, "/tmp/model.safetensors")

        self.assertTrue(torch.equal(
            model.model.layers[0].self_attn.qkv_proj.weight,
            torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]]),
        ))


if __name__ == "__main__":
    unittest.main()
