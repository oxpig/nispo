"""Derive the numbered C60 fundamental resource by geometric truncation.

P-27.1.1/Fu-3.2: truncate the regular icosahedron and follow the contiguous
vertex spiral from any pentagon about a C5 axis. No naming targets enter this
resource builder. The generated labelled graph is checked against Fig. 1.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt,atan2
from pathlib import Path


def add(a,b):return tuple(x+y for x,y in zip(a,b))
def scale(a,k):return tuple(x*k for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def subtract(a,b):return add(a,scale(b,-1))


def cyclic(points,adjacency,i):
    normal=points[i];first=subtract(points[adjacency[i][0]],normal)
    first=subtract(first,scale(normal,dot(first,normal)/dot(normal,normal)))
    second=cross(normal,first)
    return tuple(sorted(adjacency[i],key=lambda j:atan2(dot(subtract(points[j],normal),second)/sqrt(dot(second,second)),
        dot(subtract(points[j],normal),first)/sqrt(dot(first,first)))))


def cage():
    phi=(1+sqrt(5))/2
    points=[]
    for a in (-1,1):
        for b in (-1,1):points.extend(((0,a,b*phi),(a,b*phi,0),(b*phi,0,a)))
    adjacency={i:tuple(j for j,q in enumerate(points) if i!=j and abs(dot(subtract(p,q),subtract(p,q))-4)<1e-8)
        for i,p in enumerate(points)}
    assert len(points)==12 and all(len(row)==5 for row in adjacency.values())
    rotations={i:cyclic(points,adjacency,i) for i in adjacency}
    directed=tuple((i,j) for i,row in adjacency.items() for j in row);indices={edge:i for i,edge in enumerate(directed)}
    vertices=tuple(scale(add(scale(points[i],2),points[j]),1/3) for i,j in directed)
    edges=set()
    for i,j in directed:
        a=indices[i,j];edges.add(tuple(sorted((a,indices[j,i]))))
        order=rotations[i];k=order.index(j)
        for other in (order[(k-1)%5],order[(k+1)%5]):edges.add(tuple(sorted((a,indices[i,other]))))
    neighbors={i:tuple(sorted(j if i==a else a for a,j in edges if i in (a,j))) for i in range(60)}
    assert len(edges)==90 and all(len(ns)==3 for ns in neighbors.values())
    rotations60={i:cyclic(vertices,neighbors,i) for i in neighbors};paths=[]
    for ico,around in rotations.items():
        for start in range(5):
            for step in (-1,1):
                path=[indices[ico,around[(start+step*k)%5]] for k in range(5)]
                # Determine which local turn follows the supplied pentagon.
                order=rotations60[path[1]];hand=(order.index(path[2])-order.index(path[0]))%3
                while len(path)<60:
                    prev,current=path[-2:];order=rotations60[current];pos=order.index(prev)
                    options=(order[(pos+hand)%3],order[(pos-hand)%3]);following=next((v for v in options if v not in path),None)
                    if following is None:break
                    path.append(following)
                if len(path)==60:paths.append(tuple(path))
    return vertices,tuple(sorted(edges)),tuple(paths)


def numbered():
    coords,edges,paths=cage()
    graphs=set()
    for path in paths:
        numbers={idx:j+1 for j,idx in enumerate(path)}
        graphs.add(tuple(sorted(tuple(sorted((numbers[a],numbers[b]))) for a,b in edges)))
    if len(paths)!=120 or len(graphs)!=1:
        raise ValueError(f'The complete pentagon spiral family is not proved: {len(paths)} paths / {len(graphs)} labelled graphs.')
    result=next(iter(graphs))
    source_edges=((1,5),(1,9),(2,12),(7,8),(13,14),(22,23),(32,33),(56,60))
    if not all(edge in result for edge in source_edges):raise ValueError('The derived spiral disagrees with published Fig. 1 edges.')
    return dict(parent='C60-Ih',atom_count=60,edges=result,
        source='P-27.1.1/P-27.3; IUPAC Fullerene Recommendations 2002 Fu-3.2 Fig. 1 systematic numbering',
        source_url='https://iupac.qmul.ac.uk/fullerene/Fu01t03.html#f1',
        derivation='Geometric truncation of a regular icosahedron; all 120 pentagon-started contiguous vertex spirals produce the same labelled graph.',
        source_edge_checks=source_edges,spiral_frame_count=len(paths))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=numbered();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(atom_count=data['atom_count'],edge_count=len(data['edges']),spiral_frame_count=data['spiral_frame_count'],source_edges=data['source_edge_checks'])))
