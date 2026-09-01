#!/usr/bin/env python3
"""
Molecular Descriptors
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED

def calculate_properties(smiles):
    """Calculate standard molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': round(Descriptors.TPSA(mol), 2),
        'QED': round(QED.qed(mol), 3)
    }


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except:
        return smiles


def validate_smiles(smiles):
    """Check if SMILES string is chemically valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_molecular_formula(smiles):
    """Get molecular formula from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.rdMolDescriptors.CalcMolFormula(mol)
    except:
        return None


def check_substructure(query_smiles, reference_smiles):
    """Check if reference is a substructure of query"""
    try:
        query_mol = Chem.MolFromSmiles(query_smiles)
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if query_mol is None or ref_mol is None:
            return False
        return query_mol.HasSubstructMatch(ref_mol)
    except:
        return False


def compare_properties(props1, props2, tolerances=None):
    """
    Compare two property dictionaries.
    Returns fraction of properties within tolerance.
    """
    if not props1 or not props2:
        return 0.0

    if tolerances is None:
        tolerances = {
            'LogP': 0.5,
            'MW': 50,
            'HBD': 1,
            'HBA': 1,
            'TPSA': 20,
            'QED': 0.1
        }

    matched = 0
    total = 0

    for key in props2.keys():
        if key not in props1:
            continue
        total += 1
        tol = tolerances.get(key, 0)
        if abs(props1[key] - props2[key]) <= tol:
            matched += 1

    return matched / total if total > 0 else 0.0


def get_molecule_complexity(smiles):
    """Calculate a complexity score for a molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        props = calculate_properties(smiles)
        if not props:
            return 0

        n_atoms = mol.GetNumHeavyAtoms()
        score = (n_atoms * 13 + int(props['MW'] * 7) + int(abs(props['LogP']) * 11)) % 100
        return score
    except:
        return 0

