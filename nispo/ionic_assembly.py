"""P-77 binary assembly with separate ion identity, charge and stoichiometry."""
from collections import Counter
from dataclasses import dataclass
from .constants import MULT,ROOTS
from .formatting import COMPLEX_MULT,prefix_alphabetization_key,enclose_name_unit
from .ion_multipliers import needs_enclosed_ion_multiplier
from .ionic_assembly_assessment import (
    IonicComponentEvidence,
    assess_binary_ionic_components,
)


@dataclass(frozen=True,slots=True)
class IonicComponent:
    name: str
    charge: int
    identity: str
    organic: bool
    substituted: bool = False


def partial_carboxylate_name(mol,Chem):
    from .acid_salt_components import acid_salt_component
    component = acid_salt_component(mol, Chem)
    if component is not None:
        return component.name
    from .extended_inventory import selection_inventory
    from .naming_methods import _linear_order
    groups=selection_inventory(mol).groups
    anions=[g for g in groups if g.kind=='carboxylate']
    acids=[g for g in groups if g.kind=='acid']
    if len(groups)!=2 or len(anions)!=1 or len(acids)!=1:return ''
    if any(a.IsInRing() or a.GetIsotope() or int(a.GetChiralTag()) or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):return ''
    if any(int(b.GetStereo()) for b in mol.GetBonds()):return ''
    if sum(a.GetFormalCharge() for a in mol.GetAtoms())!=-1:return ''
    anion,acid=anions[0],acids[0]
    carbon={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6}-{acid.center}
    if not 2<=len(carbon)<=40 or len(carbon) not in ROOTS:return ''
    path=_linear_order(mol,carbon,(anion.center,acid.owner))
    if not path or any(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()!=1 for a,b in zip(path,path[1:])):return ''
    if set(anion.atoms)|set(acid.atoms)|carbon!=set(range(mol.GetNumAtoms())):return ''
    if len(path)==2:return 'carboxyacetate'
    return str(len(path))+'-carboxy'+ROOTS[len(path)]+'anoate'


def binary_ionic_assembly(fragments,Chem,engine,*,source_atom_maps=None,
                          expected_source_atom_ids=None):
    fragments=tuple(fragments)
    charges=[sum(a.GetFormalCharge() for a in f.GetAtoms()) for f in fragments]
    if not charges or any(not charge for charge in charges) or sum(charges)!=0:return None
    if any(a.GetNumRadicalElectrons() for f in fragments for a in f.GetAtoms()):return None
    scoped_local_ids=source_atom_maps is None
    if scoped_local_ids:
        # Direct component callers do not have a source-molecule map. Keep
        # these identifiers explicitly component-scoped; the public rule path
        # supplies the actual original atom ids below.
        source_atom_maps=tuple(
            tuple((fragment_index, atom.GetIdx()) for atom in fragment.GetAtoms())
            for fragment_index, fragment in enumerate(fragments)
        )
    else:
        source_atom_maps=tuple(tuple(atom_ids) for atom_ids in source_atom_maps)
    if (len(source_atom_maps)!=len(fragments)
            or any(len(atom_ids)!=fragment.GetNumAtoms() for atom_ids,fragment in zip(source_atom_maps,fragments))
            or any(not atom_ids for atom_ids in source_atom_maps)
            or any(
                not (type(atom_id) is int or scoped_local_ids and
                     isinstance(atom_id,tuple) and len(atom_id)==2 and
                     all(type(part) is int for part in atom_id))
                for atom_ids in source_atom_maps for atom_id in atom_ids
            )):
        return ("",(),"P-77-pending-invalid-component-partition")
    source_atom_sets=tuple(frozenset(atom_ids) for atom_ids in source_atom_maps)
    flattened_source_ids=tuple(atom_id for atom_ids in source_atom_maps for atom_id in atom_ids)
    if len(flattened_source_ids)!=len(set(flattened_source_ids)):
        return ("",(),"P-77-pending-invalid-component-partition")
    if expected_source_atom_ids is None:
        expected_source_atom_ids=frozenset().union(*source_atom_sets)
    else:
        expected_source_atom_ids=frozenset(expected_source_atom_ids)
    from .fragment_rules import (single_atom_cation_name,single_atom_halide_anion_name,
        common_single_atom_anion_name,common_polyatomic_fragment_name,simple_fragment_name)
    components=[]
    evidence=[]
    for fragment_index,(fragment,charge) in enumerate(zip(fragments,charges,strict=True)):
        typed=None
        annotated=(bool(fragment.GetStereoGroups()) or
                   any(a.GetIsotope() or int(a.GetChiralTag()) for a in fragment.GetAtoms()) or
                   any(int(b.GetStereo()) for b in fragment.GetBonds()))
        if charge>0:
            from .bluebook_ionic_acid_methods import diazonium_ion, acid_derived_cation_ion, heteroacid_derived_cation_ion
            typed=(diazonium_ion(fragment,Chem) or acid_derived_cation_ion(fragment,Chem)
                   or heteroacid_derived_cation_ion(fragment,Chem))
            if typed is None:
                from .amine_cation_components import amine_cation_component
                typed=amine_cation_component(fragment,Chem)
            if typed is None:
                from .carbon_cation_components import carbon_cation_component
                typed=carbon_cation_component(fragment,Chem)
            if typed is None:
                from .chalcogen_cation_components import chalcogen_cation_proposal
                typed=chalcogen_cation_proposal(fragment,Chem)
            if typed is not None:
                heavy=frozenset(a.GetIdx() for a in fragment.GetAtoms()
                                if a.GetAtomicNum()>1)
                if (typed.charge!=charge or typed.atoms!=heavy
                        or sum(a.GetFormalCharge() for a in fragment.GetAtoms())!=charge):
                    return None
                name=typed.name
            else:
                if annotated:return None
                from .bluebook_lambda_amine_methods import aryl_aminium_decision
                from .bluebook_hydride_assembly_methods import parent_ring_cation_decision
                aromatic_aminium = aryl_aminium_decision(fragment, Chem)
                ring_cation = parent_ring_cation_decision(fragment, Chem)
                name=(aromatic_aminium.name if aromatic_aminium is not None else '') or (
                    ring_cation.name if ring_cation is not None else '') or (
                    single_atom_cation_name(fragment,Chem) or simple_fragment_name(fragment,Chem))
        else:
            from .acid_salt_components import acid_salt_component
            typed=acid_salt_component(fragment,Chem)
            if typed is None:
                from .bluebook_ionic_acid_methods import hydroxy_chalcogen_anion_ion
                typed=hydroxy_chalcogen_anion_ion(fragment,Chem)
            if typed is None:
                from .sulfur_anion_components import sulfur_anion_component
                typed=sulfur_anion_component(fragment,Chem)
            if typed is None:
                from .retained_hydroxy_anion_components import retained_hydroxy_anion_component
                retained=retained_hydroxy_anion_component(fragment,Chem)
                if retained.status=='complete' and retained.component is not None:
                    candidate=retained.component
                    heavy=frozenset(a.GetIdx() for a in fragment.GetAtoms()
                                    if a.GetAtomicNum()>1)
                    if (not candidate.full_coverage or candidate.atoms!=heavy
                            or candidate.charge!=charge):
                        return None
                    typed=candidate
            if typed is None:
                from .polyol_anion_components import polyol_anion_component
                typed=polyol_anion_component(fragment,Chem)
                if typed is None:
                    from .polyolate_anion_components import polyolate_anion_component
                    typed=polyolate_anion_component(fragment,Chem)
                if typed is not None:
                    heavy=frozenset(a.GetIdx() for a in fragment.GetAtoms()
                                    if a.GetAtomicNum()>1)
                    if (typed.atoms | typed.embedded_atoms != heavy
                            or typed.atoms & typed.embedded_atoms):
                        return None
            if typed is None:
                from .carbanion_components import carbanion_component
                typed=carbanion_component(fragment,Chem)
                if typed is not None:
                    heavy=frozenset(a.GetIdx() for a in fragment.GetAtoms()
                                    if a.GetAtomicNum()>1)
                    if typed.atoms != heavy:return None
            if typed is None:
                from .nitrogen_anion_components import (carbon_aminide_component,
                    nitrogen_aminide_component, nitrogen_iminide_component)
                typed=nitrogen_aminide_component(fragment,Chem)
                if typed is None:
                    typed=carbon_aminide_component(fragment,Chem)
                if typed is None:
                    typed=nitrogen_iminide_component(fragment,Chem)
                if typed is not None:
                    heavy=frozenset(a.GetIdx() for a in fragment.GetAtoms()
                                    if a.GetAtomicNum()>1)
                    if typed.atoms != heavy:return None
            from .bluebook_general_methods import isotopic_formate_decision
            labelled_formate = isotopic_formate_decision(fragment,Chem) if annotated and typed is None else None
            if annotated and typed is None and labelled_formate is None:return None
            if typed is not None and typed.charge != charge:return None
            from .bluebook_general_methods import (hydride_addition_anion_decision,
                modified_carboxylate_decision, boron_cyanatido_anion_decision)
            hydride_anion = (hydride_addition_anion_decision(fragment,Chem)
                             or modified_carboxylate_decision(fragment,Chem)
                             or boron_cyanatido_anion_decision(fragment,Chem)) if typed is None else None
            name=(typed.name if typed is not None else '') or (labelled_formate.name if labelled_formate else '') or (
                  hydride_anion.name if hydride_anion is not None else '') or (
                  partial_carboxylate_name(fragment,Chem) or single_atom_halide_anion_name(fragment)
                  or common_single_atom_anion_name(fragment) or common_polyatomic_fragment_name(fragment,Chem))
        name=name or engine.name(fragment)
        if not name:return None
        identity_graph=Chem.Mol(fragment)
        for atom in identity_graph.GetAtoms():
            atom.SetAtomMapNum(0)
        identity=Chem.MolToSmiles(identity_graph,canonical=True,isomericSmiles=True)
        organic=any(a.GetAtomicNum()==6 for a in fragment.GetAtoms())
        components.append(IonicComponent(name,charge,identity,organic,
                                         substituted=bool(getattr(typed,'substituted',False))))
        component_atoms=source_atom_sets[fragment_index]
        evidence.append(IonicComponentEvidence(
            name=name,charge=charge,identity=identity,organic=organic,
            source_atom_ids=component_atoms,named_atom_ids=component_atoms,
        ))
    assessment=assess_binary_ionic_components(
        evidence,expected_source_atom_ids=expected_source_atom_ids,
    )
    if assessment.status!="admit":
        # A P-77 binary-name uncertainty must stop the generic fragment
        # assembler from emitting an unproved mixed-ion citation.
        return ("",tuple(components),"P-77-binary-pending:"+assessment.reason)
    counts=Counter(components)
    def term(component):
        count=counts[component];name=component.name
        if count==1:return name
        # P-16.3.2(c): a substituted component needs bis/tris even when
        # retained-parent locant omission removes every punctuation clue.
        complex_unit=(component.substituted or component.charge>0 and component.organic
                      or component.charge < 0 and needs_enclosed_ion_multiplier(name)
                      or any(c.isdigit() or c in '-()[]{}' for c in name))
        multiplier=(COMPLEX_MULT if complex_unit else MULT).get(count)
        if not multiplier:return ''
        return multiplier+enclose_name_unit(name) if complex_unit else multiplier+name
    terms=[]
    for positive in (True,False):
        for component in sorted((c for c in counts if (c.charge>0)==positive),key=lambda c:prefix_alphabetization_key(c.name)):
            text=term(component)
            if not text:return None
            terms.append(text)
    return ' '.join(terms),tuple(components),'P-77.1.1/P-77.2.1/P-77.3.1'
