import unittest
from pathlib import Path

try:
    from .alignment_support import model_spec, require_model
except ImportError:
    from alignment_support import model_spec, require_model


ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
  try:
    from PIL import Image

    from gemma import config
    from gemma import tokenizer
    from gemma.gemma3_preprocessor import gemma3_input_preprocessor
    from gemma.siglip_vision import pan_and_scan
    from gemma.siglip_vision import preprocessor as vision_preprocessor
  except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "install project dependencies first")
class DeterministicAlignmentTest(unittest.TestCase):

    def test_gemma3_1b_configuration(self):
        model_config = config.get_model_config("1b", dtype="float32")
        self.assertEqual(model_config.architecture, config.Architecture.GEMMA_3)
        self.assertEqual(model_config.vocab_size, 262144)
        self.assertEqual(model_config.num_hidden_layers, 26)
        self.assertEqual(model_config.get_dtype(), torch.float32)

    def test_tokenizer_special_token_contract(self):
        model_config = config.get_model_config("1b")
        model_tokenizer = tokenizer.Tokenizer(
            str(ROOT / model_config.tokenizer)
        )
        token_ids = model_tokenizer.encode(
            "The capital of Italy is", bos=True, eos=True
        )
        self.assertEqual(token_ids[0], model_tokenizer.bos_id)
        self.assertEqual(token_ids[-1], model_tokenizer.eos_id)
        self.assertEqual(
            model_tokenizer.decode(token_ids[1:-1]),
            "The capital of Italy is",
        )

    def test_pan_and_scan_covers_landscape_and_portrait(self):
        landscape = Image.new("RGB", (1024, 512), color=(128, 64, 32))
        portrait = Image.new("RGB", (512, 1024), color=(32, 64, 128))
        self.assertEqual(len(pan_and_scan.pan_and_scan(landscape)), 2)
        self.assertEqual(len(pan_and_scan.pan_and_scan(portrait)), 2)

    def test_image_preprocessing_shape_and_range(self):
        image_path = ROOT / "scripts/images/test_image.jpg"
        with Image.open(image_path) as image:
            processed = vision_preprocessor.preprocess_images_for_siglip_vision(
                [image], image_size=32
            )
        self.assertEqual(tuple(processed[0].shape), (3, 32, 32))
        self.assertGreaterEqual(processed[0].min().item(), -1.0)
        self.assertLessEqual(processed[0].max().item(), 1.0)

    def test_multimodal_preprocessor_preserves_text_and_image(self):
        image_path = ROOT / "scripts/images/test_image.jpg"
        with Image.open(image_path) as image:
            processed = gemma3_input_preprocessor(
                ["Caption this image.", image.copy()]
            )
        self.assertEqual(processed[0], "Caption this image.")
        self.assertIsInstance(processed[1], torch.Tensor)
        self.assertEqual(processed[1].shape[0], 3)

    def test_gemma4_text_backbone_forward(self):
        from gemma.gemma4.gemma4_config import Gemma4TextConfig
        from gemma.gemma4.gemma4_model import Gemma4ForCausalLM

        model_config = Gemma4TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=5,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            max_position_embeddings=32,
            layer_types=(
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
            ),
        )
        model = Gemma4ForCausalLM(model_config).eval()
        logits = model(torch.randint(0, 128, (1, 4)))
        self.assertEqual(tuple(logits.shape), (1, 4, 128))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gemma4_checkpoint_config_matches_text_backbone(self):
        from gemma.gemma4.gemma4_config import Gemma4TextConfig

        model_dir = require_model(model_spec("gemma-4-E2B-it"))
        model_config = Gemma4TextConfig.from_json(
            model_dir / "config.json"
        )
        self.assertEqual(model_config.num_hidden_layers, 35)
        self.assertEqual(model_config.layer_head_dim(0), 256)
        self.assertEqual(model_config.layer_head_dim(4), 512)
        self.assertEqual(model_config.hidden_size_per_layer_input, 256)

    def test_gemma4_vision_patch_encoder(self):
        from gemma.gemma4.gemma4_vision import Gemma4VisionConfig, Gemma4VisionModel

        vision_config = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=16,
            patch_size=2,
            position_embedding_size=16,
        )
        vision = Gemma4VisionModel(vision_config).eval()
        patches = torch.rand(1, 9, 12)
        positions = torch.tensor([[[i % 3, i // 3] for i in range(9)]])
        output = vision(patches, positions, output_length=1)
        self.assertEqual(tuple(output.shape), (1, 1, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_gemma4_image_processor_patch_contract(self):
        from gemma.gemma4.gemma4_processor import Gemma4ImageProcessor

        processed = Gemma4ImageProcessor(max_soft_tokens=1)(
            Image.new("RGB", (64, 48), color=(128, 64, 32))
        )
        self.assertEqual(tuple(processed["pixel_values"].shape), (1, 9, 768))
        self.assertEqual(tuple(processed["pixel_position_ids"].shape), (1, 9, 2))
        self.assertEqual(processed["num_soft_tokens"], 1)

    def test_gemma4_vision_config_matches_checkpoint(self):
        from gemma.gemma4.gemma4_vision import Gemma4VisionConfig

        model_dir = require_model(model_spec("gemma-4-E2B-it"))
        config = Gemma4VisionConfig.from_json(
            model_dir / "config.json"
        )
        self.assertEqual(config.num_hidden_layers, 16)
        self.assertEqual(config.position_embedding_size, 10240)
        self.assertEqual(config.pooling_kernel_size, 3)

    def test_gemma4_multimodal_forward_inserts_image_tokens(self):
        from gemma.gemma4.gemma4_config import Gemma4TextConfig
        from gemma.gemma4.gemma4_model import Gemma4ForConditionalGeneration
        from gemma.gemma4.gemma4_vision import Gemma4VisionConfig

        text_config = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            layer_types=("sliding_attention",),
        )
        vision_config = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=16,
            patch_size=2,
            position_embedding_size=16,
        )
        model = Gemma4ForConditionalGeneration(text_config, vision_config).eval()
        input_ids = torch.randint(0, 64, (1, 2))
        patches = torch.rand(1, 9, 12)
        positions = torch.tensor([[[i % 3, i // 3] for i in range(9)]])
        mask = torch.tensor([[True, False]])
        logits = model(input_ids, patches, positions, mask)
        self.assertEqual(tuple(logits.shape), (1, 2, 64))

    def test_gemma4_audio_forward(self):
        from gemma.gemma4.gemma4_audio import Gemma4AudioConfig, Gemma4AudioModel

        audio_config = Gemma4AudioConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            output_proj_dims=24,
            subsampling_conv_channels=(8, 4),
        )
        audio = Gemma4AudioModel(audio_config, feature_size=8).eval()
        features = torch.rand(1, 16, 8)
        output = audio(features)
        self.assertEqual(tuple(output.shape), (1, 4, 24))
        self.assertTrue(torch.isfinite(output).all())

    def test_gemma4_audio_processor_contract(self):
        from gemma.gemma4.gemma4_processor import Gemma4AudioProcessor

        features = Gemma4AudioProcessor()(torch.zeros(16000))
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[-1], 128)
        self.assertGreater(features.shape[1], 0)

    def test_gemma4_tokenizer_json_contract(self):
        from gemma.gemma4.gemma4_tokenizer import Gemma4Tokenizer

        model_tokenizer = Gemma4Tokenizer(
            require_model(model_spec("gemma-4-E2B-it"))
        )
        token_ids = model_tokenizer.encode(
            "The capital of Italy is", bos=True, eos=True
        )
        self.assertEqual(token_ids[0], model_tokenizer.bos_id)
        self.assertEqual(token_ids[-1], model_tokenizer.eos_id)
        self.assertEqual(
            model_tokenizer.decode(token_ids[1:-1]),
            "The capital of Italy is",
        )

    def test_gemma4_greedy_generation(self):
        from gemma.gemma4.gemma4_config import Gemma4TextConfig
        from gemma.gemma4.gemma4_model import Gemma4ForConditionalGeneration

        config = Gemma4TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=4, global_head_dim=8,
            layer_types=("sliding_attention",),
        )
        model = Gemma4ForConditionalGeneration(config).eval()
        input_ids = torch.randint(0, 32, (1, 2))
        output = model.generate(input_ids, max_new_tokens=2)
        self.assertEqual(tuple(output.shape), (1, 4))

    def test_gemma4_multimodal_audio_token_insertion(self):
        from gemma.gemma4.gemma4_audio import Gemma4AudioConfig
        from gemma.gemma4.gemma4_config import Gemma4TextConfig
        from gemma.gemma4.gemma4_model import Gemma4ForConditionalGeneration

        text_config = Gemma4TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, global_head_dim=16,
            layer_types=("sliding_attention",),
        )
        audio_config = Gemma4AudioConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            output_proj_dims=32, subsampling_conv_channels=(8, 4),
        )
        model = Gemma4ForConditionalGeneration(
            text_config, audio_config=audio_config
        ).eval()
        logits = model(
            torch.randint(0, 64, (1, 4)),
            audio_features=torch.rand(1, 16, 8),
            audio_token_mask=torch.tensor([[True, True, True, True]]),
        )
        self.assertEqual(tuple(logits.shape), (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
