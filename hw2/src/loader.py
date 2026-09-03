import os
from dataclasses import dataclass


@dataclass
class RawDoc:
    doc_id: str    
    text: str


def load_documents(data_dir: str) -> list[RawDoc]:
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        doc_id = os.path.splitext(fname)[0]
        docs.append(RawDoc(doc_id=doc_id, text=text))
    return docs


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = load_documents(os.path.join(here, "data"))
    for d in docs:
        print(f"[{d.doc_id}] {len(d.text)} String")
