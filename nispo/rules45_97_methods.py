"""Full-mode source-rule families with whole-graph eligibility proofs."""
from importlib import import_module

# Each family independently accounts for senior groups and all original
# atoms. A decline preserves the existing parent/method comparison.
_FAMILIES=('acid_halides','amide_nitrogen','esters_salts','sulfur_acids','chains_locants')


def select(mol,Chem,numbers):
    for family in _FAMILIES:
        decision=import_module('.rules45_97_'+family,__package__).select(mol,Chem,numbers)
        if decision is not None:
            return decision
    return None


def same_parent_aryl_writing(mol,Chem,parent_atoms,centers):
    """P-16 writing on an already chosen physical monoacid parent.

    This does not admit a new parent or close its preference frontier. It
    reuses the complete original-atom branch proof only when the suffix
    centres and carbon parent are identical to the pending general proposal.
    """
    from collections import Counter
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .rules45_97_chains_locants import simple_aryl_branch
    from .substituent_multiplication import simple_component_has_internal_numeric_locants
    parent_atoms=frozenset(parent_atoms)
    if (not parent_atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6
            or mol.GetAtomWithIdx(i).IsInRing() for i in parent_atoms)):
        return None
    identities=[];seen=set()
    for host in parent_atoms:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root=atom.GetIdx()
            if root in parent_atoms or root in seen or not atom.GetIsAromatic():
                continue
            atoms=frozenset(collect_subtree(mol,root,host))
            if atoms&parent_atoms or not simple_aryl_branch(mol,root,host,atoms):
                continue
            text,owned=mapped_branch(mol,Chem,root,host)
            if (not text or owned!=atoms or not simple_component_has_internal_numeric_locants(text)):
                continue
            identities.append(Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),
                rootedAtAtom=root,canonical=True,isomericSmiles=True))
            seen.update(atoms)
    if not any(count>1 for count in Counter(identities).values()):
        return None
    from .mapped_monoacid_chain import mapped_monoacid_chain_decision
    proof=mapped_monoacid_chain_decision(mol,Chem)
    if (proof is None or proof.principal_centers!=frozenset(centers)
            or parent_atoms!={i for i,_loc in proof.atom_locants if mol.GetAtomWithIdx(i).GetAtomicNum()==6}):
        return None
    return proof
