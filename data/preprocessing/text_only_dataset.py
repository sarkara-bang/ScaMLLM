"""
Text-Only Dataset for Ablation Study
Removes image modality to evaluate text-only performance
"""
import os
import json
import logging
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Sequence
import datasets
from datasets import load_dataset
import transformers

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are now working as an excellent expert in chemistry and molecule discovery.\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)


class TextOnlyDataProcessor:
    """Text-only data processor (no image processing)"""
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_length: int,
        scaffold_mapping_file: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load scaffold mapping for scaffold loss support
        self.scaffold_mapping = {}
        if scaffold_mapping_file and os.path.exists(scaffold_mapping_file):
            with open(scaffold_mapping_file, 'r') as f:
                self.scaffold_mapping = json.load(f)
            logger.info(f"[Text-Only] Loaded {len(self.scaffold_mapping)} scaffold mappings")

    def extract_scaffold_from_output(self, output_text: str) -> Optional[str]:
        """Extract scaffold SMILES from output text"""
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
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

        # Extract scaffold SMILES (for scaffold loss)
        scaffold_smiles = None
        if scaffold_image_path and scaffold_image_path in self.scaffold_mapping:
            scaffold_smiles = self.scaffold_mapping[scaffold_image_path]
            if not isinstance(scaffold_smiles, str):
                from rdkit import Chem
                if isinstance(scaffold_smiles, Chem.Mol):
                    scaffold_smiles = Chem.MolToSmiles(scaffold_smiles)

        # Fallback: extract from output
        if not scaffold_smiles:
            scaffold_smiles = self.extract_scaffold_from_output(output)

        result['scaffold_smiles'] = scaffold_smiles

        result['pixel_values'] = None

        return result


def build_text_only_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    model_max_length: int,
    scaffold_mapping_file: str = None,
    data_cache_dir=None,
    preprocessing_num_workers=None,
):
    """Build text-only dataset (no images)"""
    logger.info("[Text-Only Ablation] Building text-only dataset...")

    all_datasets = []

    # Create processor
    data_processor = TextOnlyDataProcessor(
        tokenizer=tokenizer,
        max_seq_length=model_max_length,
        scaffold_mapping_file=scaffold_mapping_file,
    )

    # Process files
    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))

        cache_path = os.path.join(data_cache_dir, f"text_only_{os.path.basename(file).split('.')[0]}")
        os.makedirs(cache_path, exist_ok=True)

        try:
            # Try loading from cache
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'[Text-Only] Dataset-{file} loaded from cache')
        except Exception:
            # Load raw dataset
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)

            # Process function
            def process_function(examples):
                results = {
                    'input_ids': [],
                    'labels': [],
                    'pixel_values': [],
                    'scaffold_smiles': []
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
                    results['pixel_values'].append(processed['pixel_values'])  # Always None
                    results['scaffold_smiles'].append(processed['scaffold_smiles'])

                return results

            # Process dataset
            processed_dataset = raw_dataset.map(
                process_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=raw_dataset['train'].column_names,
                load_from_cache_file=False,
                desc="[Text-Only] Processing dataset"
            )['train']

            # Save to cache
            processed_dataset.save_to_disk(cache_path)
            logger.info(f'[Text-Only] Dataset saved to {cache_path}')

        all_datasets.append(processed_dataset)

    # Concatenate if multiple files
    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return datasets.concatenate_datasets(all_datasets)


@dataclass
class TextOnlyDataCollator:
    """Data collator for text-only training"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                   for key in ("input_ids", "labels"))

        # Pad sequences
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

        # Attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # NO IMAGES - pixel_values is always None
        pixel_values = None

        # Collect scaffold SMILES (for scaffold loss if enabled)
        scaffold_smiles = [instance.get('scaffold_smiles', None) for instance in instances]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,  # Always None
            'scaffold_smiles': scaffold_smiles,
        }
