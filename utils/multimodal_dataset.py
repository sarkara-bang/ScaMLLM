import os
import json
import logging
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Sequence
import datasets
from datasets import load_dataset
import transformers
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.imagemol_vision import ImageMolImageProcessor

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are now working as an excellent expert in chemistry and molecule discovery.\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)


class EnhancedMultiModalDataProcessor:
    """Data processor with scaffold SMILES extraction"""
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        max_seq_length: int,
        image_dir: str = None,
        scaffold_mapping_file: str = None,
    ):
        self.tokenizer = tokenizer
        if image_processor is None:
            self.image_processor = ImageMolImageProcessor()
        else:
            self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.image_dir = image_dir

        self.scaffold_mapping = {}
        if scaffold_mapping_file and os.path.exists(scaffold_mapping_file):
            with open(scaffold_mapping_file, 'r') as f:
                self.scaffold_mapping = json.load(f)
            logger.info(f"Loaded {len(self.scaffold_mapping)} scaffold mappings")

    def extract_smiles_from_text(self, text: str) -> Optional[str]:
        """Extract SMILES from text"""
        if not text or not isinstance(text, str):
            return None

        import re
        patterns = [r'"([^"]+)"', r'is:\s*"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1)
                mol = Chem.MolFromSmiles(candidate)
                if mol:
                    return candidate
        return None

    def extract_scaffold_from_output(self, output_text: str) -> Optional[str]:
        """Extract scaffold SMILES from output text"""
        import re
        patterns = [r'"([^"]+)"', r'is:\s*"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']

        for pattern in patterns:
            match = re.search(pattern, output_text)
            if match:
                smiles = match.group(1)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    try:
                        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                        if scaffold_mol:
                            return Chem.MolToSmiles(scaffold_mol)
                    except:
                        pass
        return None

    def process_example(self, example):
        """Process single example"""
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        scaffold_image_path = example.get('scaffold_image', None)

        if input_text:
            instruction = instruction + '\n' + input_text

        source = PROMPT_TEMPLATE.format_map({'instruction': instruction})
        target = f"{output}{self.tokenizer.eos_token}"

        tokenized_source = self.tokenizer(source, return_attention_mask=False)
        tokenized_target = self.tokenizer(target, return_attention_mask=False, add_special_tokens=False)

        input_ids = tokenized_source['input_ids'] + tokenized_target['input_ids']
        labels = [IGNORE_INDEX] * len(tokenized_source['input_ids']) + tokenized_target['input_ids']

        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        result = {
            'input_ids': input_ids,
            'labels': labels,
        }

        target_smiles = self.extract_smiles_from_text(output)
        result['target_smiles'] = target_smiles

        scaffold_smiles = None
        if scaffold_image_path and scaffold_image_path in self.scaffold_mapping:
            scaffold_smiles = self.scaffold_mapping[scaffold_image_path]
            if isinstance(scaffold_smiles, Chem.Mol):
                scaffold_smiles = Chem.MolToSmiles(scaffold_smiles)

        if not scaffold_smiles:
            scaffold_smiles = self.extract_scaffold_from_output(output)

        result['scaffold_smiles'] = scaffold_smiles

        if scaffold_image_path:
            if self.image_dir and not os.path.isabs(scaffold_image_path):
                full_image_path = os.path.join(self.image_dir, scaffold_image_path)
            else:
                full_image_path = scaffold_image_path

            try:
                image = Image.open(full_image_path).convert('RGB')
                image_features = self.image_processor(images=image, return_tensors="pt")
                if isinstance(image_features, dict):
                    result['pixel_values'] = image_features['pixel_values'][0]
                else:
                    result['pixel_values'] = image_features.pixel_values[0]
            except Exception as e:
                logger.warning(f"Error processing image {scaffold_image_path}: {e}")
                result['pixel_values'] = None
        else:
            result['pixel_values'] = None

        return result


def build_multimodal_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    image_processor,
    model_max_length: int,
    image_dir: str = None,
    scaffold_mapping_file: str = None,
    data_cache_dir=None,
    preprocessing_num_workers=None,
):
    """Build multimodal dataset"""
    logger.info("Building multimodal dataset...")

    all_datasets = []

    data_processor = EnhancedMultiModalDataProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_seq_length=model_max_length,
        image_dir=image_dir,
        scaffold_mapping_file=scaffold_mapping_file,
    )

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))

        cache_path = os.path.join(data_cache_dir, f"enhanced_multimodal_{os.path.basename(file).split('.')[0]}")
        os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'Dataset-{file} loaded from cache')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, field='samples', cache_dir=cache_path)

            def process_function(examples):
                results = {
                    'input_ids': [],
                    'labels': [],
                    'pixel_values': [],
                    'scaffold_smiles': [],
                    'target_smiles': []
                }

                for i in range(len(examples['instruction'])):
                    example = {
                        'instruction': examples['instruction'][i],
                        'input': examples['input'][i],
                        'output': examples['output'][i],
                        'scaffold_image': examples.get('scaffold_image', [None] * len(examples['instruction']))[i]
                    }

                    processed = data_processor.process_example(example)
                    results['input_ids'].append(processed['input_ids'])
                    results['labels'].append(processed['labels'])
                    results['pixel_values'].append(processed['pixel_values'])
                    results['scaffold_smiles'].append(processed['scaffold_smiles'])
                    results['target_smiles'].append(processed['target_smiles'])

                return results

            processed_dataset = raw_dataset.map(
                process_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=raw_dataset['train'].column_names,
                load_from_cache_file=False,
                desc="Processing multimodal dataset"
            )['train']

            processed_dataset.save_to_disk(cache_path)
            logger.info(f'Dataset saved to {cache_path}')

        all_datasets.append(processed_dataset)

    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return datasets.concatenate_datasets(all_datasets)


@dataclass
class MultiModalDataCollator:
    """Data collator for multimodal batches"""
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[any] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                   for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in labels],
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        pixel_values = [instance['pixel_values'] for instance in instances]
        if any(x is not None for x in pixel_values):
            pixel_values = torch.stack([
                torch.tensor(x) if (x is not None and not isinstance(x, torch.Tensor))
                else (x if x is not None else torch.zeros((3, 224, 224)))
                for x in pixel_values
            ])
        else:
            pixel_values = None

        scaffold_smiles = [instance.get('scaffold_smiles', None) for instance in instances]
        target_smiles = [instance.get('target_smiles', None) for instance in instances]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'scaffold_smiles': scaffold_smiles,
            'target_smiles': target_smiles,
        }
