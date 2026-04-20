import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import transformers

from utils.util import ensure_bert_checkpoint


class PlantSegDataset(data.Dataset):
    def __init__(
        self,
        args,
        split="train",
        image_transforms=None,
        max_tokens=20,
        eval_mode=False,
        logger=None,
    ) -> None:
        self.args = args
        self.split = split
        self.image_transforms = image_transforms
        self.max_tokens = max_tokens
        self.eval_mode = eval_mode
        self.root = Path(args.plantseg_root)
        manifest_path = self.root / "main.json"
        self.samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples = [sample for sample in self.samples if sample["split"] == split]

        bert_path = ensure_bert_checkpoint(logger, "./checkpoints/bert-base-uncased")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_path)

        if logger:
            logger.info(
                f"=> loaded successfully 'plantseg', split {split}, size {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def _encode_text(self, text):
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(text=text, add_special_tokens=True)
        input_ids = input_ids[: self.max_tokens]
        padded_input_ids[: len(input_ids)] = input_ids
        attention_mask[: len(input_ids)] = [1] * len(input_ids)
        return torch.tensor(padded_input_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        mask_path = self.root / sample["mask"]
        caption = sample["caption"][self.args.caption_index]

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_array = (np.array(mask) > 0).astype(np.uint8)
        target = Image.fromarray(mask_array, mode="L")
        embedding, attention_mask = self._encode_text(caption)

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, target)

        if self.eval_mode:
            return img, target, embedding, attention_mask, sample["id"], sample["mask"]
        return img, target, embedding, attention_mask
