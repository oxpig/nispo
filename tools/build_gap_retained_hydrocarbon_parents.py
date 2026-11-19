"""Build the finite primary Table 2.7 hydride vocabulary, never panel data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from audit_gap_retained_hydrocarbons import PARENTS
from check_release_acceptance import opsin,canonical


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=project/'nispo/resources/bluebook_retained_hydrocarbon_parents.json')
    args=parser.parse_args();source=project/'nispo/resources/opsin_retained_mono_fused_ring_templates.json'
    records=json.loads(source.read_text());references=opsin(PARENTS);parents=[]
    for name,reference in zip(PARENTS,references,strict=True):
        if reference is None:raise AssertionError('A source-named parser reference is unavailable: '+name)
        bare=name.partition('H-')[2] if 'H-' in name else name
        aliases={name,bare,bare[:-1] if bare.endswith('e') else bare}
        choices=[r for r in records if r['name'] in aliases and canonical(Chem.MolFromSmiles(r['smiles']))==canonical(reference)]
        if not choices:raise AssertionError('No graph/H-compatible original resource frame for '+name)
        choices.sort(key=lambda row:(row['name']!=name,row['name']!=bare,row['name']))
        record=choices[0];graph=Chem.MolFromSmiles(record['smiles']);mapping={int(i):loc for i,loc in record['locants'].items()}
        if (graph is None or graph.GetNumAtoms()!=len(mapping) or set(mapping)!=set(range(graph.GetNumAtoms()))
                or any(a.GetAtomicNum()!=6 or a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms())):
            raise AssertionError('The complete neutral carbon frame is not proved: '+name)
        if 'H-' in name:
            site=name.partition('H-')[0];atoms=[graph.GetAtomWithIdx(i) for i,loc in mapping.items() if loc==site]
            if len(atoms)!=1 or atoms[0].GetIsAromatic() or atoms[0].GetTotalNumHs()!=2:
                raise AssertionError('The source indicated-H methylene state is not proved: '+name)
        parents.append(dict(name=name,smiles=canonical(graph),formula=CalcMolFormula(graph),
            independent_cycle_rank=graph.GetNumBonds()-graph.GetNumAtoms()+1,
            original_frame_smiles=record['smiles'],original_locants=record['locants'],original_resource_name=record['name']))
    result=dict(rule='P-25.1.1/Table 2.7/P-25.7',source='https://iupac.qmul.ac.uk/BlueBook/P2.html#P-25.1.1',
        scope='Exact complete unmodified neutral carbon hydrides only; isotope/configuration and all derivatives require their own comparisons.',
        graph_provenance='Existing OPSIN-derived located frames, matched to OPSIN 2.9 references from the independently selected primary vocabulary; this is not independent validation of every published drawing.',
        source_resource_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parents=parents)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(parents=len(parents),output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
