from datasets import load_dataset
import json
import os

def load_and_save_dataset():
    dataset_dict = load_dataset(
        "rag-datasets/rag-mini-wikipedia",
        "text-corpus"
    )

    dataset = dataset_dict["passages"]

    os.makedirs("faiss_index", exist_ok=True)

    passages = dataset["passage"][:]   # ✅ convert to list

    data = {
        "passages": passages
    }

    with open("faiss_index/metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Dataset saved successfully.")

if __name__ == "__main__":
    load_and_save_dataset()