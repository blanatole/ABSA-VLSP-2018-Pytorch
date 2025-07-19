"""
Data Processing Module for Vietnamese ABSA
Ported from TensorFlow to PyTorch implementation
"""

import re
import os
import csv
import emoji
import urllib
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from io import StringIO
from vncorenlp import VnCoreNLP
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader


class PolarityMapping:
    """Mapping for sentiment polarities"""
    INDEX_TO_POLARITY = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
    INDEX_TO_ONEHOT = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}
    POLARITY_TO_INDEX = {None: 0, 'positive': 1, 'negative': 2, 'neutral': 3}


class VietnameseTextCleaner:
    """Vietnamese text cleaning utilities"""
    VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
    
    @staticmethod
    def remove_html(text: str) -> str:
        return re.sub(r'<[^>]*>', '', text)
    
    @staticmethod
    def remove_emoji(text: str) -> str:
        return emoji.replace_emoji(text, '')
    
    @staticmethod
    def remove_url(text: str) -> str:
        return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)
    
    @staticmethod
    def remove_email(text: str) -> str:
        return re.sub(r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+', '', text)
    
    @staticmethod
    def remove_phone_number(text: str) -> str:
        return re.sub(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$', '', text)
    
    @staticmethod
    def remove_hashtags(text: str) -> str:
        return re.sub(r'#\w+', '', text)
    
    @staticmethod
    def remove_unnecessary_characters(text: str) -> str:
        text = re.sub(fr"[^\sa-zA-Z0-9{VietnameseTextCleaner.VN_CHARS}]", ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def process_text(text: str) -> str:
        """Apply all cleaning steps"""
        text = VietnameseTextCleaner.remove_html(text)
        text = VietnameseTextCleaner.remove_emoji(text)
        text = VietnameseTextCleaner.remove_url(text)
        text = VietnameseTextCleaner.remove_email(text)
        text = VietnameseTextCleaner.remove_phone_number(text)
        text = VietnameseTextCleaner.remove_hashtags(text)
        text = VietnameseTextCleaner.remove_unnecessary_characters(text)
        return text


class VietnameseToneNormalizer:
    """Vietnamese tone normalization utilities"""
    VOWELS_TABLE = [
        ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
    ]
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Vietnamese Unicode characters"""
        import unicodedata
        return unicodedata.normalize('NFC', text)
    
    @staticmethod
    def normalize_sentence_typing(text: str, vinai_normalization: bool = False) -> str:
        """Normalize Vietnamese typing mistakes"""
        # Implementation for common Vietnamese typing mistakes
        # e.g., 'lựơng' => 'lượng', 'thỏai mái' => 'thoải mái'
        return text  # Simplified for now


class VietnameseTextPreprocessor:
    """Main Vietnamese text preprocessing class"""
    
    def __init__(self, vncorenlp_dir: str = './VnCoreNLP', 
                 extra_teencodes: Optional[Dict] = None,
                 max_correction_length: int = 512):
        self.vncorenlp_dir = vncorenlp_dir
        self.extra_teencodes = extra_teencodes or {}
        self.max_correction_length = max_correction_length
        
        self._load_vncorenlp()
        self._build_teencodes()
        self._load_error_corrector()
    
    def _load_vncorenlp(self):
        """Load VnCoreNLP for word segmentation"""
        self.word_segmenter = None
        try:
            if self._get_vncorenlp_files():
                self.word_segmenter = VnCoreNLP(
                    os.path.join(self.vncorenlp_dir, 'VnCoreNLP-1.2.jar'),
                    annotators='wseg',
                    quiet=False
                )
                print('VnCoreNLP word segmenter loaded successfully.')
            else:
                print('Failed to load VnCoreNLP word segmenter.')
        except Exception as e:
            print(f'Error loading VnCoreNLP: {e}')
    
    def _get_vncorenlp_files(self) -> bool:
        """Download VnCoreNLP files if not exists"""
        files_needed = [
            '/VnCoreNLP-1.2.jar',
            '/models/wordsegmenter/vi-vocab',
            '/models/wordsegmenter/wordsegmenter.rdr'
        ]
        
        for file_path in files_needed:
            local_path = self.vncorenlp_dir + file_path
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                download_url = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master' + file_path
                try:
                    print(f'Downloading {download_url} to {local_path}')
                    urllib.request.urlretrieve(download_url, local_path)
                except Exception as e:
                    print(f'Failed to download {download_url}: {e}')
                    return False
        return True
    
    def _build_teencodes(self):
        """Build teencode mapping dictionary"""
        # Default teencodes - can be extended
        default_teencodes = {
            'khách sạn': ['ks'],
            'nhà hàng': ['nhahang'],
            'nhân viên': ['nv'],
            'điện thoại': ['dt'],
            'facebook': ['fb', 'face']
        }
        
        self.teencodes = {**default_teencodes, **self.extra_teencodes}
    
    def _load_error_corrector(self):
        """Load Vietnamese error correction model"""
        try:
            self.corrector = pipeline(
                'text2text-generation',
                model='bmd1905/vietnamese-correction-v2',
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            print('Vietnamese error corrector loaded successfully.')
        except Exception as e:
            print(f'Failed to load error corrector: {e}')
            self.corrector = None
    
    def normalize_teencodes(self, text: str) -> str:
        """Convert teencodes to proper Vietnamese"""
        for proper, codes in self.teencodes.items():
            for code in codes:
                text = re.sub(rf'\b{re.escape(code)}\b', proper, text, flags=re.IGNORECASE)
        return text
    
    def correct_vietnamese_errors(self, texts: List[str]) -> List[str]:
        """Correct Vietnamese spelling/grammar errors"""
        if self.corrector is None:
            return texts
        
        corrected = []
        for text in texts:
            if len(text) <= self.max_correction_length:
                try:
                    result = self.corrector(text, max_length=len(text) + 50)
                    corrected.append(result[0]['generated_text'])
                except:
                    corrected.append(text)
            else:
                corrected.append(text)
        return corrected
    
    def word_segment(self, text: str) -> str:
        """Segment Vietnamese words using VnCoreNLP"""
        if self.word_segmenter is None:
            return text
        
        try:
            result = self.word_segmenter.tokenize(text)
            return ' '.join([' '.join(sentence) for sentence in result])
        except:
            return text
    
    def process_text(self, text: str, normalize_tone: bool = True, segment: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        # Basic cleaning
        text = VietnameseTextCleaner.process_text(text)
        text = text.lower()
        
        # Tone normalization
        if normalize_tone:
            text = VietnameseToneNormalizer.normalize_unicode(text)
            text = VietnameseToneNormalizer.normalize_sentence_typing(text)
        
        # Teencode normalization
        text = self.normalize_teencodes(text)
        
        # Word segmentation
        if segment and self.word_segmenter:
            text = self.word_segment(text)
        
        return text.strip()
    
    def process_batch(self, texts: List[str], correct_errors: bool = True) -> List[str]:
        """Process a batch of texts"""
        # First pass: basic preprocessing
        processed = [self.process_text(text, segment=False) for text in texts]
        
        # Error correction (optional, slow)
        if correct_errors and self.corrector:
            processed = self.correct_vietnamese_errors(processed)
        
        # Final word segmentation
        processed = [self.word_segment(text) for text in processed]
        
        return processed


class VLSP2018Parser:
    """Parser for VLSP 2018 dataset format"""
    
    def __init__(self, train_txt_path: str, val_txt_path: str = None, test_txt_path: str = None):
        self.dataset_paths = {'train': train_txt_path, 'val': val_txt_path, 'test': test_txt_path}
        self.reviews = {'train': [], 'val': [], 'test': []}
        self.aspect_categories = set()
        
        # Remove None paths
        self.dataset_paths = {k: v for k, v in self.dataset_paths.items() if v}
        self.reviews = {k: v for k, v in self.reviews.items() if k in self.dataset_paths}
        
        self._parse_input_files()
    
    def _parse_input_files(self):
        """Parse VLSP 2018 format text files"""
        print(f'[INFO] Parsing {len(self.dataset_paths)} input files...')
        
        for dataset_type, txt_path in self.dataset_paths.items():
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                review_blocks = content.strip().split('\n\n')
                
                for block in tqdm(review_blocks, desc=f'Parsing {dataset_type}'):
                    lines = block.split('\n')
                    if len(lines) < 3:
                        continue
                        
                    # Extract sentiment information
                    sentiment_info = re.findall(r'\{([^,]+)#([^,]+), ([^}]+)\}', lines[2].strip())
                    
                    review_data = {}
                    for aspect, category, polarity in sentiment_info:
                        aspect_category = f'{aspect.strip()}#{category.strip()}'
                        self.aspect_categories.add(aspect_category)
                        review_data[aspect_category] = PolarityMapping.POLARITY_TO_INDEX[polarity.strip()]
                    
                    self.reviews[dataset_type].append((lines[1].strip(), review_data))
        
        self.aspect_categories = sorted(self.aspect_categories)
        print(f'[INFO] Found {len(self.aspect_categories)} aspect categories')
    
    def txt2csv(self):
        """Convert parsed data to CSV format"""
        print('[INFO] Converting parsed data to CSV files...')
        
        for dataset_type, txt_path in self.dataset_paths.items():
            csv_path = txt_path.replace('.txt', '.csv')
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Review'] + self.aspect_categories)
                
                for review_text, review_data in tqdm(self.reviews[dataset_type], desc=f'Writing {dataset_type} CSV'):
                    row = [review_text] + [review_data.get(aspect_category, 0) for aspect_category in self.aspect_categories]
                    writer.writerow(row)
            
            print(f'[INFO] Saved {dataset_type} data to {csv_path}')


class VLSP2018Dataset(Dataset):
    """PyTorch Dataset for VLSP 2018 ABSA data"""
    
    def __init__(self, csv_path: str, tokenizer, aspect_categories: List[str], 
                 preprocessor: VietnameseTextPreprocessor = None, 
                 max_length: int = 256, approach: str = 'multitask'):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.aspect_categories = aspect_categories
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.approach = approach
        
        # Load data
        self.data = pd.read_csv(csv_path)
        print(f'[INFO] Loaded {len(self.data)} samples from {csv_path}')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get review text
        review_text = row['Review']
        
        # Preprocess text if preprocessor available
        if self.preprocessor:
            review_text = self.preprocessor.process_text(review_text)
        
        # Tokenize
        encoding = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels based on approach
        labels = []
        for aspect_category in self.aspect_categories:
            label = row.get(aspect_category, 0)
            labels.append(label)
        
        if self.approach == 'multitask':
            # Flatten one-hot encoding for multitask approach
            flatten_labels = []
            for label in labels:
                flatten_labels.extend(PolarityMapping.INDEX_TO_ONEHOT[label])
            labels = torch.tensor(flatten_labels, dtype=torch.float)
        else:
            # Keep as separate labels for multibranch approach
            labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def create_dataloaders(train_csv: str, val_csv: str, test_csv: str, 
                      tokenizer, aspect_categories: List[str],
                      preprocessor: VietnameseTextPreprocessor = None,
                      batch_size: int = 32, max_length: int = 256,
                      approach: str = 'multitask', num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test"""
    
    datasets = {}
    for split, csv_path in [('train', train_csv), ('val', val_csv), ('test', test_csv)]:
        if csv_path and os.path.exists(csv_path):
            datasets[split] = VLSP2018Dataset(
                csv_path=csv_path,
                tokenizer=tokenizer,
                aspect_categories=aspect_categories,
                preprocessor=preprocessor,
                max_length=max_length,
                approach=approach
            )
    
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders.get('train'), dataloaders.get('val'), dataloaders.get('test')


if __name__ == '__main__':
    # Example usage
    print("Testing VLSP2018 data processing...")
    
    # Test parser
    parser = VLSP2018Parser(
        train_txt_path='../data/datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.txt',
        val_txt_path='../data/datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.txt',
        test_txt_path='../data/datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.txt'
    )
    
    print(f"Found aspect categories: {parser.aspect_categories}")
    
    # Convert to CSV
    parser.txt2csv()
    print("Data processing completed!") 