import json
import os
import pickle
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


def build_lid_index_splits(
    index_path: str = "./data/lid_index.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Build train/test splits from an existing lid index (stratified by language).
    Writes lid_index_train.pkl and lid_index_test.pkl in the same directory as index_path.
    Same structure as the full index: dict mapping (tar_number, filename) -> lang.
    """
    with open(index_path, "rb") as f:
        index = pickle.load(f)

    keys = list(index.keys())
    labels = [index[k] for k in keys]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(keys, labels))

    lid_index_train = {keys[i]: index[keys[i]] for i in train_idx}
    lid_index_test = {keys[i]: index[keys[i]] for i in test_idx}

    base = Path(index_path).parent
    train_path = base / "lid_index_train.pkl"
    test_path = base / "lid_index_test.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(lid_index_train, f, protocol=4)
    with open(test_path, "wb") as f:
        pickle.dump(lid_index_test, f, protocol=4)

    print(f"Saved {train_path} ({len(lid_index_train)} entries)")
    print(f"Saved {test_path} ({len(lid_index_test)} entries)")


def build_lid_index(index_path: str = "./data/lid_index.pkl", hf_token: str = None):
    """
    Build a language ID index from the JSONL results file.
    The index maps (tar_number, filename) to predicted language.
    Saves the index as a pickle file.

    Args:
        index_path (str): Path to save the index pickle file.
    """

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")

    lid_folder = Path(index_path).parent

    if not os.path.exists(lid_folder):
        os.makedirs(lid_folder)

    if not os.path.exists(lid_folder / "lang_id_results.jsonl"):
        # Download the results file from Hugging Face
        import requests

        url = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/lang_id_results.jsonl"
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download lid_results.jsonl: {response.status_code}"
            )
        with open(lid_folder / "lang_id_results.jsonl", "wb") as f:
            f.write(response.content)
        print(f"Downloaded  lang_id_results.jsonl to {lid_folder / 'lang_id_results.jsonl'}")

    index = {}

    with open(lid_folder / "lang_id_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tar_number = obj["tar_number"]
            filename = os.path.basename(obj["filepath"])
            lang = obj["prediction"]

            index[(tar_number, filename)] = lang

    print(f"Built index with {len(index)} entries")

    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=4)

    print(f"Saved to {index_path}")

    build_lid_index_splits(index_path)
