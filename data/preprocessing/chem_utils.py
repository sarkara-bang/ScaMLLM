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
        'HBA': Lipinski.NumHAcceptors(mol)
    }

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except:
        return smiles
def validate_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_molecular_formula(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.rdMolDescriptors.CalcMolFormula(mol)
    except:
        return None


def check_substructure(query_smiles, reference_smiles):
    try:
        query_mol = Chem.MolFromSmiles(query_smiles)
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if query_mol is None or ref_mol is None:
            return False
        return query_mol.HasSubstructMatch(ref_mol)
    except:
        return False


def compare_properties(props1, props2, tolerances=None):

    if not props1 or not props2:
        return 0.0

    if tolerances is None:
        tolerances = {'LogP': 0.5,'MW': 50,'HBD': 1,'HBA': 1}

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


from rdkit_utils import canonicalize
normalize_smiles = canonicalize

__all__ = [
    'calculate_properties',
    'canonicalize_smiles',
    'validate_smiles',
    'get_molecular_formula',
    'check_substructure',
    'compare_properties',
    'get_molecule_complexity',
    'normalize_smiles',
    'canonicalize'
]
