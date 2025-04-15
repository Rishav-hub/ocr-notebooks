# -*- coding: utf-8 -*-
"""Table model pytorch training loop_hyperparameter_optimized_row_column_data

Original file is located at
    https://colab.research.google.com/drive/1arV1zQ2IAkDH9dF_UAOvO7bkXpruPQHk
"""

import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed
import os
import gc
import re
import json
from datetime import datetime
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_scheduler
)
from datasets import Dataset
from datasets import Image as ds_img
from polyleven import levenshtein
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from evaluate import load
from prettytable import PrettyTable
from transformers import PreTrainedTokenizerBase, PreTrainedModel

cer_metric = load("cer" , trust_remote_code=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Ensure consistent hashing (important in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✅ Seed set to {seed} across all libraries.")

# Example usage
seed_everything(43)
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

#give the base path of the directory which should have the IMG and JSON files, ensure the corresponding files must have same file names
base_path = Path("/mnt/win_share1/XELPMOC_2025/P2_(Im_Ke)")
metadata_path = base_path.joinpath("KEY")
image_path = base_path.joinpath("IMG")

free_memory()

class CFG:
    # General
    debug = False
    num_proc = 1
    num_workers = 0
    gpus = 1
    model_folder = "/mnt/win_share1/XELPMOC_2025/Model/UB/sprint_4/pytorch_15_April_Table_RC"
    os.makedirs(model_folder, exist_ok=True)
    checkpoint_name = f"""ub_Table_RC_15Apr_pytorch_{datetime.today().strftime("%Y_%m_%d")}"""

    # Data
    max_length = 1024
    image_height = 1280
    image_width = 960

    # Training
    epochs = 5
    val_check_interval = 1.0  # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    lr = 3e-5
    lr_scheduler_type = "cosine"
    num_warmup_steps = 300
    seed = 42
    output_path = "output"
    log_steps = 200
    batch_size = 5
    use_wandb = False
    weight_decay = 0.01

def json2token(obj, sort_json_key: bool = True):
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += (
                    fr"<{k}>"
                    + obj[k]
                    + fr"</{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join([json2token(item, sort_json_key) for item in obj])
    else:
        obj = f"<{str(obj)}/>"
        return obj

PROMPT_TOKEN = "<s>"
END_PROMPT = "</s>"
def process_dataset(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
        text = PROMPT_TOKEN + json2token(data) + END_PROMPT
        # text = json.dumps(data)
        if image_path.joinpath(f"{file_name.stem}.tiff").is_file():
            img_path = str(image_path.joinpath(f"{file_name.stem}.tiff"))
            return {"id":file_name.stem, "image_path":img_path, "text":text}
        else:
            return {"id":file_name.stem, "image_path":"", "text": "", }

free_memory()

def gen_data(files: List[Union[str, os.PathLike]]) -> Dict[str, str]:
    """
    This function takes a list of json files and returns a generator that yields a
    dictionary with the ground truth string and the path to the image.

    Args:
        files (list): A list of json files

    Returns:
        generator: A generator that yields a dictionary with the ground truth string and
            the path to the corresponding image.
    """
    c = 0
    for f in metadata_path.glob("*.json"):
        # if c == 500:
        #     break
        # c+=1
        yield {
            **process_dataset(f),
            }


ds = Dataset.from_generator(
    gen_data, gen_kwargs={"files": metadata_path}, num_proc=CFG.num_proc
)

print(f"The overall size of the dataset is {len(ds)}")

new_tokens = set()
task_start_token = PROMPT_TOKEN
blank_token = '[BLANK]'

for index, file_name in enumerate(metadata_path.glob("*.json")):
    if index <=2000:
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
            keys = data.keys()
        for row in data.keys():
            new_tokens.add(f"<{row}>")
            new_tokens.add(f"</{row}>")

        # Extract XML-like tags from row values
        for row_value in data.values():
            tags = [tag.split(">")[0] + ">" for tag in row_value.split("<") if ">" in tag]  # Extract tag names
            for tag in tags:
                clean_tag = tag.replace(">", "").strip()
                new_tokens.add(f"<{clean_tag}>")
                new_tokens.add(f"<{clean_tag}>")

# Convert the set back to a list if necessary
new_tokens = list(new_tokens)

print(f"The number of new tokens added to the vocabulary are: {len(new_tokens)}")

# ds = ds.select(range(200))

# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.tokenizer.add_special_tokens({"additional_special_tokens":  [task_start_token] + new_tokens})
# processor.tokenizer.add_tokens([seperator_token] ) #+ ['[BLANK]']
processor.tokenizer.add_tokens('[BLANK]') #+ ['[BLANK]']
# processor.tokenizer.add_tokens(["<one>"])
processor.image_processor.size = [960,1280]

config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = (CFG.image_height, CFG.image_width)
config.decoder.max_length = CFG.max_length

print(CFG.image_height, CFG.image_width, CFG.max_length)

def check_for_unk(examples: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Check for unknown tokens in the given examples.

    This function takes a dictionary containing a list of ground truth texts and
    tokenizes them using the processor's tokenizer. It then checks for any unknown
    tokens in the tokenized text and returns a dictionary containing a list of the
    unknown tokens for each example.

    Args:
        examples (dict): A dictionary containing a list of ground truth texts.
            Example: {"ground_truth": ["text1", "text2", ...]}

    Returns:
        dict: A dictionary containing a list of unknown tokens for each example.
            Example: {"unk_tokens": [["unk1", "unk2"], [], ["unk3"], ...]}
    """

    texts = examples["text"]

    ids = processor.tokenizer(texts).input_ids
    tokens = [processor.tokenizer.tokenize(x, add_special_tokens=True) for x in texts]

    unk_tokens = []
    for example_ids, example_tokens in zip(ids, tokens):
        example_unk_tokens = []
        for i in range(len(example_ids)):
            if example_ids[i] == processor.tokenizer.unk_token_id:
                example_unk_tokens.append(example_tokens[i])
        unk_tokens.append(example_unk_tokens)

    return {"unk_tokens": unk_tokens}

unk = ds.map(check_for_unk, batched=True, num_proc=CFG.num_proc)

# Let's look at only the examples with unknown tokens
# unk = unk.filter(lambda x: len(x["unk_tokens"]) > 0, num_proc=CFG.num_proc)
# print(len(unk))
# unk[0]

all_unk_tokens = [x for y in unk["unk_tokens"] for x in y]
Counter(all_unk_tokens)

# example_str = "0.1 1 1990"
# temp_ids = processor.tokenizer(example_str).input_ids
# print("ids:", temp_ids)
# print("tokenized:", processor.tokenizer.tokenize(example_str))
# print("decoded:", processor.tokenizer.decode(temp_ids))
# print("unk id:", processor.tokenizer.unk_token_id)

# Adding these tokens should mean that there should be very few unknown tokens
num_added = processor.tokenizer.add_tokens(["<one>"] + new_tokens)
print(num_added, "tokens added")

config.pad_token_id = processor.tokenizer.pad_token_id
config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([PROMPT_TOKEN])[0]
config.eos_token_id = processor.tokenizer.convert_tokens_to_ids([END_PROMPT])[0]

one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
unk_token_id = processor.tokenizer.unk_token_id

def replace_unk_tokens_with_one(example_ids: List[int], example_tokens: List[str], one_token_id:int, unk_token_id:int) -> List[int]:
    """
    Replace unknown tokens that represent "1" with the correct token id.

    Args:
        example_ids (list): List of token ids for a given example
        example_tokens (list): List of tokens for the same given example
        one_token_id (int): Token id for the "<one>" token
        unk_token_id (int): Token id for the unknown token

    Returns:
        list: The updated list of token ids with the correct token id for "1"
    """

    temp_ids = []
    for id_, token in zip(example_ids, example_tokens):
        if id_ == unk_token_id and token == "1":
            id_ = one_token_id
        temp_ids.append(id_)
    return temp_ids

def preprocess(examples: Dict[str, str], processor: DonutProcessor, CFG: CFG) -> Dict[str, Union[torch.Tensor, List[int], List[str]]]:
    """
    Preprocess the given examples.

    This function processes the input examples by tokenizing the texts, replacing
    any unknown tokens that represent "1" with the correct token id, and loading
    the images.

    Args:
        examples (dict): A dictionary containing ground truth texts, image paths, and ids
        processor: An object responsible for tokenizing texts and processing images
        CFG: A configuration object containing settings and hyperparameters

    Returns:
        dict: A dictionary containing preprocessed images, token ids, and ids
    """
    pixel_values = []
    texts = examples["text"]
    ids = processor.tokenizer(
        texts,
        add_special_tokens=False,
        max_length=CFG.max_length,
        padding=True,
        truncation=True,
    ).input_ids

    if isinstance(texts, str):
        texts = [texts]

    tokens = [processor.tokenizer.tokenize(text, add_special_tokens=False) for text in texts]

    one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
    unk_token_id = processor.tokenizer.unk_token_id
    pad_token_id = processor.tokenizer.pad_token_id
    
    final_ids = [
        replace_unk_tokens_with_one(example_ids, example_tokens, one_token_id, unk_token_id)
        for example_ids, example_tokens in zip(ids, tokens)
    ]

    for sample in examples["image_path"]:
        sample = np.array(sample.convert("RGB"))
        pixel_values.append(processor(sample, random_padding=True).pixel_values)

    return {
        "pixel_values": torch.tensor(np.vstack(pixel_values)),
        "input_ids": final_ids,
        "id": examples["id"],
        "gt_text": examples["text"],
    }


# image_ds = ds.cast_column("image_path", ds_img())
# image_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG))
# sample = image_ds[[0, 1, 2]]
# print(sample["pixel_values"].shape)
# print(processor.decode(sample["input_ids"][2]))
# print(len(sample["input_ids"][2]))
# print(processor.tokenizer.convert_ids_to_tokens(sample["input_ids"][2]))

split_dataset = ds.train_test_split(test_size=0.04)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

train_dataset = train_dataset.cast_column("image_path", ds_img())
train_dataset.set_transform(partial(preprocess, processor=processor, CFG=CFG))

val_dataset = val_dataset.cast_column("image_path", ds_img())
val_dataset.set_transform(partial(preprocess, processor=processor, CFG=CFG))

print(f"Train Dataset has {len(train_dataset)} images")
print(f"Validation dataset has {len(val_dataset)} images")

def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List[int], str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for DataLoader.

    This function takes a list of samples and combines them into a batch with
    properly padded input_ids.

    Args:
        samples (List[Dict[str, Union[torch.Tensor, List[int], str]]]):
            A list of samples, where each sample is a dictionary containing
            "pixel_values" (torch.Tensor), "input_ids" (List[int]), and "id" (str).

    Returns:
        Dict[str, Union[torch.Tensor, List[str]]]:
            A dictionary containing the combined pixel values, padded input_ids, and ids.
    """

    batch = {}
    pad_token_id = processor.tokenizer.pad_token_id
    batch["pixel_values"] = torch.stack([x["pixel_values"] for x in samples])

    max_length = max([len(x["input_ids"]) for x in samples])

    # Make a multiple of 8 to efficiently use the tensor cores
    if max_length % 8 != 0:
        max_length = (max_length // 8 + 1) * 8

    input_ids = [
        x["input_ids"] + [pad_token_id] * (max_length - len(x["input_ids"]))
        for x in samples
    ]

    labels = torch.tensor(input_ids)
    labels[labels == pad_token_id] = -100 # ignore loss on padding tokens
    batch["labels"] = labels

    batch["id"] = [x["id"] for x in samples]
    batch['gt_text'] = [x['gt_text'] for x in samples]

    return batch

CFG.debug = False

if CFG.debug:
    train_ds = train_dataset.select(range(100))
    val_ds = val_dataset.select(range(5))
else:
    train_ds = train_dataset
    val_ds = val_dataset

train_dataloader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
)
val_dataloader = DataLoader(
    val_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
)

num_training_steps = len(train_dataloader) * CFG.epochs // CFG.gpus
batch = next(iter(train_dataloader))
batch.keys(), [(k, v.shape) for k, v in batch.items() if k not in ["id" ,"gt_text"]]

def clean_ada(out):
    # out = re.sub(r"<.*?>", "", out, count=1).strip()
    out = out.replace("<s>", "")
    out = out.replace("</s>", "")
    return out

def sanity_check(dataloader, index=0):
    """
    Function to visualize an image from the dataloader along with its ground truth text.

    Args:
    - dataloader: PyTorch DataLoader object
    - index (int): Index of the image within the batch to visualize

    Returns:
    - Displays the image and prints the ground truth text
    """
    # Get one batch from the dataloader
    for batch in dataloader:
        pixel_values = batch["pixel_values"]  # Shape: (batch_size, channels, height, width)
        gt_text = batch["gt_text"]  # Ground truth text
        break  # Take only the first batch

    # Ensure the index is within bounds
    batch_size = pixel_values.shape[0]
    if index >= batch_size:
        raise ValueError(f"Index {index} out of range for batch size {batch_size}")

    # Select the image and corresponding ground truth text
    image_tensor = pixel_values[index]  # Shape: (C, H, W)
    gt_output = gt_text[index]

    print(f"Image Tensor Shape: {image_tensor.shape}")
    print(f"Ground Truth Text: {gt_output}")

    # Convert tensor to NumPy (C, H, W) -> (H, W, C)
    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Normalize if necessary
    # image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())  # Normalize to [0, 1]

    # Display the image
    plt.figure(figsize=(12, 8))  # Adjust figure size
    plt.imshow(image_array)
    plt.axis("off")
    plt.show()

sanity_check(train_dataloader, index=4)

def validation_metrics(pred_output, gt_output):
    gt_output = processor.token2json(clean_ada(gt_output))
    pred_output = processor.token2json(clean_ada(pred_output))
    all_metrics = []
    key_metrics = {}

    gt_keys = []
    pred_keys = []
    for gt_key in gt_output:
        gt_keys.append(gt_key)
    for pred_key in pred_output:
        pred_keys.append(pred_key)

    for k in gt_output:
        if k in pred_keys:
            pred_str = pred_output[k]
        else:
            pred_str = ""
        gt_str = gt_output[k]

        try:
            if gt_str == "":
                gt_str = "[BLANK]"
            if pred_str == "":
                pred_str = "[BLANK]"
            metric = cer_metric.compute(predictions=[pred_str], references=[gt_str])
            key_metrics[k] = metric
            all_metrics.append(metric)
        except Exception as e:
            # print(e) #enable to see mis macthes in gt and prediction

            # print([pred_output], [gt_output])
            # print([pred_str], [gt_str])
            key_metrics[k] = 1.0
            all_metrics.append(1.0)

            pass
        # print(f"CER for key {k} : {metric}")
    average_metric = sum(all_metrics)/len(all_metrics)
    return average_metric, key_metrics



import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

# Load model from huggingface.co
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"New embedding size: {new_emb}")
# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1] # (height, width)
model.config.decoder.max_length = 1024

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([PROMPT_TOKEN])[0]
model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids([END_PROMPT])[0]


# model.load_state_dict(torch.load("/content/drive/MyDrive/001_projects/FSL/FSL_2025/Model/UB/sprint_4/pytorch_19_march/ub_19th_mar_pytorch_2025_03_27_4.pth", map_location=torch.device("cuda")))  # Adjust path

free_memory()

from transformers import get_cosine_schedule_with_warmup

def train_donut_model(model: PreTrainedModel, processor: PreTrainedTokenizerBase,
                      train_loader: DataLoader, val_loader: DataLoader,
                      num_epochs: int, lr: float, device: str):
    """
    Train the Donut model using native PyTorch.

    Args:
        model (PreTrainedModel): The pretrained Donut model.
        processor (PreTrainedTokenizerBase): The tokenizer/processor for the model.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (str): Device to use ('cuda' or 'cpu').
    """
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    best_val_loss = float('inf')
    num_warmup_steps = int(0.1 * num_training_steps)  # Adjust as needed
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip_val)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        total_cer = []
        val_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                gt_sequences = batch["gt_text"]
                batch_size = pixel_values.shape[0]

                decoder_input_ids = torch.full(
                    (batch_size, 1), model.config.decoder_start_token_id, device=device
                )

                outputs = model(pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=CFG.max_length,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

                for pred_seq, gt_seq in zip(processor.tokenizer.batch_decode(outputs.sequences), gt_sequences):
                    pred_seq = pred_seq.replace(processor.tokenizer.eos_token, "").replace("<one>", "1")
                    gt_seq = gt_seq.replace(processor.tokenizer.eos_token, "")
                    cer, _ = validation_metrics(pred_seq, gt_seq)
                    total_cer.append(cer)

                    # Print only the first 3 examples
                    if sample_count < 3:

                        print(f"\nSample {sample_count + 1}:")
                        print(f"🔹 Predicted: {pred_seq}")
                        print(f"🔹 Ground Truth: {gt_seq}")
                        sample_count += 1

        avg_cer = sum(total_cer) / len(total_cer)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation CER: {avg_cer:.4f}")

        # Save model checkpoint based on best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CFG.model_folder, f"{CFG.checkpoint_name}_{epoch}.pth"))
            print("Best model checkpoint saved.")

        # 🟢 Push to Hugging Face Hub after each epoch
        print("🔄 Pushing model and processor to Hugging Face Hub...")
        model.push_to_hub("Laskari-Naveen/pytorch_8April_Table_RC", commit_message=f"Update after Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation CER: {avg_cer:.4f}", create_pr=1)
        processor.push_to_hub("Laskari-Naveen/pytorch_8April_Table_RC", commit_message=f"Update after Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation CER: {avg_cer:.4f}", create_pr=1)
        print("✅ Model and processor pushed successfully!")

    print("Training complete.")

free_memory()

num_epochs = 6  # Adjust as needed
learning_rate = 5e-5  # Adjust as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
train_donut_model(model, processor, train_dataloader, val_dataloader, num_epochs, learning_rate, device)

# 🟢 Push to Hugging Face Hub after each epoch
print("🔄 Pushing model and processor to Hugging Face Hub...")
model.push_to_hub("Laskari-Naveen/pytorch_8April_Table_RC", commit_message=f"Last commit {num_epochs}", create_pr=1)
processor.push_to_hub("Laskari-Naveen/pytorch_8April_Table_RC", commit_message=f"Last commit {num_epochs}", create_pr=1)
print("✅ Model and processor pushed successfully!")
free_memory()
