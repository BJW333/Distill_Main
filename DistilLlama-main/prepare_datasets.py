import os
from datasets import load_dataset

def prepare_wikitext(output_dir="wikitext/wikitext-103-raw-v1"):
    """
    Downloads the Wikitext-103 raw splits and saves them as .raw files.
    """
    # Load the dataset using Hugging Face's datasets library
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to a .raw file
    split_map = {"train": "wiki.train.raw", "validation": "wiki.valid.raw", "test": "wiki.test.raw"}
    for hf_split, filename in split_map.items():
        out_path = os.path.join(output_dir, filename)
        print(f"Saving Wikitext {hf_split} to {out_path}…")
        with open(out_path, "w", encoding="utf-8") as f:
            for example in ds[hf_split]:
                # Replace newlines to keep a clean format
                text = example["text"].replace("\n", " ")
                f.write(text + "\n")


def prepare_openwebtext(output_path="openwebtext/subsets/openwebtext.txt"):
    """
    Downloads the OpenWebText dataset and saves it as a single text file.
    """
    ds = load_dataset("openwebtext")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving OpenWebText to {output_path}…")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds["train"]:
            text = example.get("text", example.get("content", ""))
            f.write(text.replace("\n", " ") + "\n")


if __name__ == "__main__":
    # Ensure you have installed the datasets library:
    #    pip install datasets
    prepare_wikitext()
    prepare_openwebtext()
    print("✅ Datasets downloaded and saved.")
