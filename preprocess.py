import os 
from collections import defaultdict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def flatten_2d_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

class ReliReader(object):
    """
    Reads the RELI dataset.
    """
    def __init__(self, files):
        """
        :param files: list of files to read
        """
        self.files = files
        self.title_string = "#Título"
        self.body_string = "#Corpo"
        self.book_string = "#Livro"
        self.review_id_string = "#Resenha"
        self.score_string = "#Nota"
    
    def read_lines(self):
        """
        Reads the lines from the files.
        :return: generator of (filename, line) tuples
        """
        for filename in self.files:
            with open(filename, 'r') as f:
                for line in f:
                    yield filename, line

    def must_skip_line(self, line):
        """
        Checks if the line must be skipped.
        """
        return line.startswith("[")

    def is_metadata(self, line):
        return line.startswith("#")

    def is_separator_line(self, line):
        """
        Checks if the line is a separator line.
        """
        return not line.strip()
    
    def convert_labels(self, labels):
        label_list = flatten_2d_list(labels)
        positive, negative = False, False
        for label in label_list:
            if label.strip().endswith("+"):
                positive = True
            elif label.strip().endswith("-"):
                negative = True
    
        if positive and not negative:
            return "positive"
        elif negative and not positive:
            return "negative" 
        elif positive and negative:
            return "mixed"
        else:
            return "neutral"

    def convert_buffer(self, buffer):
        """
        Converts the buffer to a sentence.
        """
        labels = []
        tokens = []
        for line in buffer:
            columns = line.split("\t")
            labels.append(columns[1:])
            tokens.append(columns[0])
        return " ".join(tokens), self.convert_labels(labels)

    def read_sentences(self):
        """
        Reads the sentences from the files.
        :return: generator of (filename, sentence) tuples
        """
        buffer = []
        previous_filename = None
        metadata_fields = {
            "source": None,
            "title": None,
            "book": None,
            "review_id": None,
            "score": None,
            "sentence_id": None,
        }
        for idx, (filename, line) in enumerate(self.read_lines()):
            
            if previous_filename is not None and filename != previous_filename:
                if buffer:
                    yield metadata_fields, self.convert_buffer(buffer)
                    buffer = []

            if self.must_skip_line(line):
                continue

            metadata_fields["sentence_id"] = idx
            metadata_fields["source"] = filename
            metadata_fields["unique_review_id"] = self.get_unique_review_id(filename, metadata_fields["book"], metadata_fields["review_id"])

            if self.is_metadata(line):
                if self.is_title(line):
                    metadata_fields["title"] = True
                elif self.is_body(line):
                    metadata_fields["title"] = False
                elif self.is_book(line):
                    metadata_fields["book"] = self.get_book(line)
                elif self.is_review_id(line):
                    metadata_fields["review_id"] = self.get_review_id(line)
                elif self.is_score(line):
                    metadata_fields["score"] = self.get_score(line)
                continue

            if self.is_separator_line(line):
                if buffer:
                    yield metadata_fields, self.convert_buffer(buffer)
                    buffer = []
            else:
                buffer.append(line)
            previous_filename = filename

        if buffer:
            yield metadata_fields, self.convert_buffer(buffer)
    
    def get_book(self, line):
        return line.replace(self.book_string, "").strip().strip("_")
    def get_review_id(self, line):
        return int(line.replace(self.review_id_string, "").strip().strip("_"))
    def get_score(self, line):
        return float(line.replace(self.score_string, "").strip().strip("_"))

    def get_unique_review_id(self, source, book, review_id):
        return f"{source[:-4]}_{book}_{review_id}"

    def is_title(self, line):
        return line.startswith(self.title_string)
    def is_body(self, line):
        return line.startswith(self.body_string)
    def is_book(self, line):
        return line.startswith(self.book_string)
    def is_review_id(self, line):
        return line.startswith(self.review_id_string)
    def is_score(self, line):
        return line.startswith(self.score_string)

def main():
    files = [ x for x in os.listdir() if x.endswith(".txt") and x.lower().startswith("reli")]
    reader = ReliReader(files)
    book_dict = defaultdict(list)
    for fields, line in reader.read_sentences():
        row = {}
        row.update(fields)
        row["sentence"] = line[0]
        row["label"] = line[1]
        book_dict[fields["source"]].append(row)
    for filename, rows in book_dict.items():
        book_dict[filename] = pd.DataFrame(rows)
    
    dev_count = int(len(book_dict) * 0.16)
    test_count = int(len(book_dict) * 0.2)

    dev_books = list(book_dict.keys())[:dev_count]
    test_books = list(book_dict.keys())[dev_count:dev_count+test_count]
    train_books = list(book_dict.keys())[dev_count+test_count:]

    train_df = pd.concat([book_dict[x] for x in train_books])
    dev_df = pd.concat([book_dict[x] for x in dev_books])
    test_df = pd.concat([book_dict[x] for x in test_books])

    logging.info(f"Train: {len(train_df)}")
    logging.info(f"Dev: {len(dev_df)}")
    logging.info(f"Test: {len(test_df)}")

    train_df.to_csv("train.csv", index=False)
    dev_df.to_csv("dev.csv", index=False)
    test_df.to_csv("test.csv", index=False)

if __name__ == "__main__":
    main()