import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from spacy.lang.de import German
from spacy.lang.en import English

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face        
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        #loading datatset from Hugging Face
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        #loading tokenizers
        # self.de_nlp = spacy.load("de_core_news_sm")
        # self.en_nlp = spacy.load("en_core_web_sm") 
        self.de_nlp = German()
        self.en_nlp =English()
        #Special tokens
        self.special_tokens = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}      
        
        if src_vocab is None or tgt_vocab is None:
            self.build_vocab()
        else :
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            
        self.process_data()
        

    # Building vocabulary for both source and target languages
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        train_data = load_dataset("bentrevett/multi30k", split = "train")

        src_words = set()
        tgt_words = set()
        for example in train_data:
            de_tokens = self.de_nlp(example['de'])
            en_tokens = self.en_nlp(example['en'])
            src_words.update([token.text.lower() for token in de_tokens])
            tgt_words.update([token.text.lower() for token in en_tokens])

        self.src_vocab = self.special_tokens.copy()
        self.tgt_vocab = self.special_tokens.copy()

        for word in src_words:
            if word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)
        for word in tgt_words:
            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)
    
    # Tokenization functions for German and English sentences
    def tokenize_de(self, sentence):
        return [token.text.lower() for token in self.de_nlp(sentence)]

    def tokenize_en(self, sentence):
        return [token.text.lower() for token in self.en_nlp(sentence)]


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        self.processed_data = []
        for example in self.dataset:
            de_tokens = self.tokenize_de(example['de'])
            en_tokens = self.tokenize_en(example['en'])
            
            src_ids = [self.special_tokens["<sos>"]]
            src_ids += [self.src_vocab.get(token, self.special_tokens["<unk>"]) for token in de_tokens]
            src_ids.append(self.special_tokens["<eos>"])

            tgt_ids = [self.special_tokens["<sos>"]]
            tgt_ids += [self.tgt_vocab.get(token, self.special_tokens["<unk>"]) for token in en_tokens]
            tgt_ids.append(self.special_tokens["<eos>"])
            self.processed_data.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
