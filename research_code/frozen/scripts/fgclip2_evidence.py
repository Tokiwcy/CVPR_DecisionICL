"""Pinned FG-CLIP 2 Base global evidence adapter (not spatial grounding)."""
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from dgdr.hashing import file_sha256

MODEL = PROJECT.parent / "models/fg-clip2-base"
REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
WEIGHT_HASH = "5328bf1538afb4b7f4aa844f749655285d0d79490f31a25aeefc5603d9b173fe"

class FGClip2Encoder:
    model_id = str(MODEL)
    model_revision = REVISION + "|weights=" + WEIGHT_HASH + "|patches256-short64-bf16-strict-cpu-init-v1"
    uncalibrated_evidence_transform = "fgclip2-global-native-scale__uncalibrated__group-softmax__one-vs-rest-log-margin-v1"

    def __init__(self, device="cuda", batch_size=1):
        self.device, self.batch_size = device, batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
        from safetensors.torch import load_model
        if file_sha256(MODEL / "model.safetensors") != WEIGHT_HASH:
            raise ValueError("FG-CLIP 2 checkpoint hash mismatch")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self._processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
        # Transformers 5 meta loading leaves this older custom model's nonpersistent
        # position_ids uninitialized and plain long-text masks on meta. Construct
        # the original model normally on CPU, then strictly load EVERY saved tensor.
        # No checkpoint/config changes or random missing-weight fallback are allowed.
        config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
            attn_implementation="sdpa")
        load_model(self._model, str(MODEL / "model.safetensors"), strict=True)
        emb = self._model.text_model.embeddings
        if not torch.equal(emb.position_ids, torch.arange(config.text_config.longtext_len).unsqueeze(0)):
            raise ValueError("FG-CLIP 2 positional buffer initialization failed")
        if emb.mask1.is_meta or emb.mask2.is_meta:
            raise ValueError("unmaterialized long-text mask")
        self._model = self._model.eval().to(self.device)

    def autocast(self):
        import torch
        from contextlib import nullcontext
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else nullcontext()

    @staticmethod
    def normalized(features):
        import torch
        result = torch.nn.functional.normalize(features.float(), dim=-1).detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite FG-CLIP 2 embedding")
        return result.astype(np.float32)

    def encode_texts(self, texts):
        import torch
        self._load()
        texts = list(texts)
        lengths = [len(ids) for ids in self._tokenizer(texts, truncation=False)["input_ids"]]
        if max(lengths, default=0) > 64:
            raise ValueError("FG-CLIP 2 short text exceeds 64 tokens; do not silently truncate")
        outputs = []
        for start in range(0, len(texts), 32):
            batch = self._tokenizer(texts[start:start + 32], padding="max_length", max_length=64,
                truncation=False, return_tensors="pt").to(self.device)
            with torch.inference_mode(), self.autocast():
                features = self._model.get_text_features(input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"), walk_type="short")
            outputs.append(self.normalized(features))
        return np.concatenate(outputs)

    def encode_images(self, image_paths):
        import torch
        from PIL import Image
        self._load()
        outputs = []
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            try:
                for path in image_paths[start:start + self.batch_size]:
                    with Image.open(path) as raw:
                        images.append(raw.convert("RGB"))
                batch = self._processor(images=images, max_num_patches=256, return_tensors="pt").to(self.device)
                with torch.inference_mode(), self.autocast():
                    features = self._model.get_image_features(**batch)
                outputs.append(self.normalized(features))
            finally:
                for image in images:
                    image.close()
        return np.concatenate(outputs)

    def logit_scale_and_bias(self):
        self._load()
        return (float(self._model.logit_scale.detach().float().exp().cpu()),
                float(self._model.logit_bias.detach().float().cpu()))

if __name__ == "__main__":
    import torch
    from dgdr.io import load_manifest
    torch.set_num_threads(4)
    rows = load_manifest(PROJECT / "artifacts/cub_full_current_atoms_20260904/cub.train.jsonl", expected_split="train")
    encoder = FGClip2Encoder()
    images = encoder.encode_images([s.image_path for s in rows[:2]])
    texts = encoder.encode_texts(["This image shows a bird with a short beak.", "This image shows a bird with a long beak."])
    scale, bias = encoder.logit_scale_and_bias()
    logits = scale * images @ texts.T + bias
    print(dict(image_shape=images.shape, text_shape=texts.shape, finite=bool(np.isfinite(logits).all()), scale=scale, bias=bias))
