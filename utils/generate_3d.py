from rdkit import Chem
from rdkit.Chem import AllChem
import os

OUTPUT_DIR = "3dmodel.py"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_3d_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status != 0:
        raise RuntimeError("3D embedding failed")

    AllChem.UFFOptimizeMolecule(mol)

    output_path = os.path.join(OUTPUT_DIR, "molecule.sdf")
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()

    return output_path