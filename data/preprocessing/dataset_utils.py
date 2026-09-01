from rdkit_utils import canonicalize
def standardize_structure(input_smiles, reference_smiles=None,
                         target_props=None, verbose=False):
    """
    Standardize molecular structures for datasets.

    Args:
        input_smiles: Input molecule SMILES
        reference_smiles: Optional reference
        target_props: Optional properties
        verbose: Enable logging

    Returns:
        Standardized SMILES string
    """
    return canonicalize(
        smiles=input_smiles,
        scaffold=reference_smiles,
        properties=target_props,
        verbose=verbose
    )


__all__ = ['standardize_structure']
