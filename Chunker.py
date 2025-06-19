import json
import hashlib
from collections import defaultdict
from transformers import AutoTokenizer
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict

class Chunker:

    def __init__(self):

        self.chunking_method = "default"
        self.chunk_size = 512
        self.chunk_overlap = 0
        self.chunk_overlap_ratio = 0.1
        self.horizontal_threshold = 100
        self.vertical_threshold = 30
        self.tokenizer_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


    @staticmethod
    def is_close_vertically(el1, el2, vertical_threshold):
        return abs(el1["bbox"]["t"] - el2["bbox"]["t"]) < vertical_threshold

    @staticmethod
    def is_close_horizontally(el1, el2, horizontal_threshold):
        l1, r1 = el1["bbox"]["l"], el1["bbox"]["r"]
        l2, r2 = el2["bbox"]["l"], el2["bbox"]["r"]
        mid1 = (l1 + r1) / 2
        mid2 = (l2 + r2) / 2
        return abs(mid1 - mid2) < horizontal_threshold

    @staticmethod
    def generate_unique_id(element):
        """Generates a unique hash based on self_ref and document fields."""
        key_fields = {
            "title": element.get("title", ""),
            "text_content": element.get("text_content", ""),
            "pages": element.get("pages", []),
            "document": element.get("document", ""),
        }
        key_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    @staticmethod
    def format_image_text(image_text, image_caption):

        if image_caption:
            return image_caption + " " + image_text
        else:
            return image_text
        
    @staticmethod
    def format_table_text(table_row, table_header_rows, table_caption):
        table_header_rows = table_header_rows if table_header_rows else []
        table_caption = table_caption or ""

        header_str = "\n".join(table_header_rows)

        return f"{table_caption}\nHeaders:\n{header_str}\nRow:\n{table_row}"

        
    def merge_chunks_by_title(self, chunks: List[Dict], token_limit=512):

        merged_chunks = []
        
        title_groups = defaultdict(list)

        for chunk in chunks:
            if "title" not in chunk:
                raise ValueError("Missing 'title' in chunk")
            title_groups[chunk["title"]].append(chunk)

        for title, chunk_group in title_groups.items():

            current_text = ""
            current_chunks = []

            for chunk in chunk_group:
                chunk_text = chunk["text_content"]
                test_text = f"{current_text} {chunk_text}".strip() if current_text else chunk_text
                token_count = len(self.tokenizer.tokenize(test_text))

                if token_count <= token_limit:
                    current_text = test_text
                    current_chunks.append(chunk)
                else:
                    if current_text:
                        # Save previous merged chunk
                        merged_chunks.append({
                            "title": title,
                            "text_content": current_text.strip(),
                            "pages": sorted(set(c["page"] for c in current_chunks)),
                        })
                    current_text = chunk_text
                    current_chunks = [chunk]

            # Add final group
            if current_text:
                merged_chunks.append({
                    "title": title,
                    "text_content": current_text.strip(),
                    "pages": sorted(set(c["page"] for c in current_chunks)),
                })

        # Final pass: split chunks that exceed token limit using RecursiveCharacterTextSplitter
        final_chunks = []
        for chunk in merged_chunks:
            tokens = self.tokenizer.tokenize(chunk["text_content"])
            if len(tokens) <= token_limit:
                final_chunks.append(chunk)
            else:
                # Split text
                splits = self.text_splitter.split_text(chunk["text_content"])
                char_idx = 0
                for split in splits:
                    # Estimate which pages this split spans using proportion of characters
                    split_len = len(split)
                    start_char = chunk["text_content"].find(split, char_idx)
                    end_char = start_char + split_len
                    char_idx = end_char  # for next search

                    # Best effort: assign all original pages since we lose granularity after merge
                    final_chunks.append({
                        "title": chunk["title"],
                        "text_content": split.strip(),
                        "pages": chunk["pages"],  # Optional: narrow this down if you preserve char ranges
                    })

        return final_chunks

    
    def text_structure_chunking(self, elements):
        
        texts = [
            el for el in elements 
            if el.get("label") not in ("picture", "table", "caption") and el.get("text_content", "").strip()
        ]
        images = [
            el for el in elements 
            if el.get("label") == "picture" and el.get("text_content", "").strip()
        ]
        tables = [
            el for el in elements 
            if el.get("label") == "table"# and el.get("text_content", "").strip(), should add this?
        ]

        texts.sort(key=lambda x: (x["page"], -x["bbox"]["t"], x["bbox"]["l"]))
        images.sort(key=lambda x: (x["page"], -x["bbox"]["t"], x["bbox"]["l"]))
        tables.sort(key=lambda x: (x["page"], -x["bbox"]["t"], x["bbox"]["l"]))

        documents = list(set([doc["document"] for doc in elements]))

        text_chunks = []
        for doc in documents:
            doc_texts = [el for el in texts if el["document"] == doc]
            doc_chunks = self.merge_chunks_by_title(doc_texts, self.chunk_size)
            for doc_chunk in doc_chunks:
                doc_chunk["document"] = doc
                doc_chunk["label"] = "text"
            text_chunks.extend(doc_chunks)
        
        image_chunks = []
        for doc in documents:
            image_texts = [el for el in images if el["document"] == doc]
            for chunk in image_texts:

                image_caption = chunk["image_metadata"]["image_caption"]
                tokens = self.tokenizer.tokenize(chunk["text_content"])
                caption_tokens = self.tokenizer.tokenize(image_caption) if image_caption else []
                n_limit, _ = divmod(len(tokens), self.chunk_size)

                #Check is meant to consider repetitions of captions per image chunk
                if len(tokens + caption_tokens * n_limit) <= self.chunk_size:
                    image_chunks.append({
                        "title": chunk["title"],
                        "text_content": self.format_image_text(chunk["text_content"], image_caption),     
                        "pages": [chunk["page"]],
                        "document" : doc,
                        "label": "picture"})
                else:
                    splits = self.text_splitter.split_text(chunk["text_content"])
                    for split in splits:
                        image_chunks.append({
                            "title": chunk["title"],
                            "text_content": self.format_image_text(split.strip(), image_caption),
                            "pages": chunk["pages"],
                            "document" : doc,
                            "label": "picture"
                        })

        table_chunks = []
        for doc in documents:
            table_texts = [el for el in tables if el["document"] == doc]
            for chunk in table_texts:

                table_title = chunk["title"]
                table_caption = chunk["table_metadata"]["caption"]
                table_content = chunk["table_metadata"]["content"]
                num_columns = chunk["table_metadata"]["num_columns"]
                table_header = chunk["table_metadata"]["header"]

                if num_columns>0:
                    table_header = [table_header[i:i + num_columns] for i in range(0, len(table_header), num_columns)]

                table_header_rows = [" | ".join(row) for row in table_header]
                table_content = [content for content in table_content if content not in table_header_rows]

                # splitting table content does not make sense
                # it is better to keep it as a whole
                # keep monitored the number of tokens
                # should add page list if table spans multiple pages TODO

                for table_row in table_content:

                    table_chunks.append({
                        "title": table_title,
                        "text_content": self.format_table_text(table_row, table_header_rows, table_caption),
                        "pages": [chunk["page"]],
                        "document" : doc,
                        "label": "table"
                    })

        chunks = text_chunks + image_chunks + table_chunks
        
        for chunk in chunks:
            chunk["id"] = self.generate_unique_id(chunk)

        return chunks