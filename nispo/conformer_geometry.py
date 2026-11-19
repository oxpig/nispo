"""Geometry behind P-94 descriptors; no molecular naming or priority guesses."""
from __future__ import annotations

from math import atan2,degrees,sqrt


def signed_torsion(points,axis_path=(),distance_tolerance=1e-8):
    import numpy as np
    x,a,b,y=np.asarray(points,dtype=float)
    axis=b-a;length=float(np.linalg.norm(axis))
    if length<=distance_tolerance:
        return None
    axis/=length
    for position in axis_path:
        offset=np.asarray(position)-a
        if np.linalg.norm(offset-np.dot(offset,axis)*axis)>distance_tolerance:
            return None
    front=x-a;rear=y-b
    front-=np.dot(front,axis)*axis;rear-=np.dot(rear,axis)*axis
    if min(np.linalg.norm(front),np.linalg.norm(rear))<=distance_tolerance:
        return None
    return float(degrees(atan2(float(np.dot(axis,np.cross(front,rear))),float(np.dot(front,rear)))))


def torsion_region(angle,tolerance):
    magnitude=abs(angle)
    for boundary,options in ((30.,('sp','sc')),(90.,('sc','ac')),(150.,('ac','ap'))):
        if abs(magnitude-boundary)<=tolerance:
            return '',options
    return ('sp' if magnitude<30 else 'sc' if magnitude<90 else 'ac' if magnitude<150 else 'ap'),()


def regular_projected_ligands(points,center,other,angular_tolerance,distance_tolerance):
    import numpy as np
    axis=np.asarray(other)-np.asarray(center);length=np.linalg.norm(axis)
    if length<=distance_tolerance:return False
    axis/=length;vectors=[]
    for point in points:
        vector=np.asarray(point)-np.asarray(center);vector-=np.dot(vector,axis)*axis
        length=np.linalg.norm(vector)
        if length<=distance_tolerance:return False
        vectors.append(vector/length)
    if len(vectors)<2:return False
    angles=sorted(degrees(atan2(float(np.dot(axis,np.cross(vectors[0],v))),float(np.dot(vectors[0],v))))%360 for v in vectors)
    gaps=[(angles[(i+1)%len(angles)]-angles[i])%360 for i in range(len(angles))]
    return max(abs(gap-360/len(angles)) for gap in gaps)<=angular_tolerance


def plane_distances(points,reference,tolerance):
    import numpy as np
    cloud=np.asarray(points,dtype=float);base=cloud[list(reference)]
    center=base.mean(axis=0)
    _,singular,vectors=np.linalg.svd(base-center,full_matrices=False)
    if len(singular)<2 or singular[1]<=tolerance:
        return None
    distances=(cloud-center)@vectors[-1]
    if np.max(np.abs(distances[list(reference)]))>tolerance:
        return None
    return distances


def six_ring_puckering(points,tolerance):
    """Cremer–Pople first-mode mean plane and six-ring Fourier coordinates."""
    import numpy as np
    cloud=np.asarray(points,dtype=float);cloud=cloud-cloud.mean(axis=0)
    j=np.arange(6,dtype=float)
    first_sine=np.sum(cloud*np.sin(2*np.pi*j/6)[:,None],axis=0)
    first_cosine=np.sum(cloud*np.cos(2*np.pi*j/6)[:,None],axis=0)
    normal=np.cross(first_sine,first_cosine)
    norm=float(np.linalg.norm(normal))
    if norm<=tolerance:
        return None
    z=cloud@(normal/norm)
    cosine=sqrt(2/6)*float(np.sum(z*np.cos(4*np.pi*j/6)))
    sine=-sqrt(2/6)*float(np.sum(z*np.sin(4*np.pi*j/6)))
    q2=sqrt(cosine*cosine+sine*sine)
    q3=sqrt(1/6)*float(np.sum(z*((-1.)**j)))
    total=sqrt(q2*q2+q3*q3)
    if total<=tolerance:
        return (total,0.,0.,q2,q3)
    return (total,float(degrees(atan2(q2,q3))),float(degrees(atan2(sine,cosine))%360),q2,q3)


def ring_shape(points,bond_orders,distance_tolerance,angular_tolerance):
    """Return a proved ideal/source plane condition or an unclassified form."""
    import numpy as np
    size=len(points);cloud=np.asarray(points,dtype=float)
    if size<3:
        return 'unclassified',(),None,()
    if plane_distances(cloud,tuple(range(size)),distance_tolerance) is not None:
        return 'planar',(),None,()
    candidates=[]
    if size==5:
        for flap in range(5):
            distances=plane_distances(cloud,tuple(i for i in range(5) if i!=flap),distance_tolerance)
            if distances is not None and abs(distances[flap])>distance_tolerance:
                candidates.append(('envelope',(flap,)))
    puckering=six_ring_puckering(cloud,distance_tolerance) if size==6 else None
    if size==6:
        for first in range(3):
            second=first+3
            distances=plane_distances(cloud,tuple(i for i in range(6) if i not in {first,second}),distance_tolerance)
            if distances is not None and min(abs(distances[first]),abs(distances[second]))>distance_tolerance:
                candidates.append(('chair' if distances[first]*distances[second]<0 else 'boat',(first,second)))
        double=[i for i,order in enumerate(bond_orders) if order==2]
        if len(double)==1 and all(order in {1,2} for order in bond_orders):
            first=(double[0]+3)%6;second=(double[0]+4)%6
            distances=plane_distances(cloud,tuple(i for i in range(6) if i not in {first,second}),distance_tolerance)
            if (distances is not None and min(abs(distances[first]),abs(distances[second]))>distance_tolerance
                    and distances[first]*distances[second]<0):
                candidates.append(('half-chair',(first,second)))
        if puckering is not None and all(order==1 for order in bond_orders):
            total,theta,phi,_q2,_q3=puckering
            phase=abs(((phi-30+30)%60)-30)
            if total>distance_tolerance and abs(theta-90)<=angular_tolerance and phase<=angular_tolerance:
                candidates.append(('twist',()))
    if size>=8 and size%2==0 and all(order==1 for order in bond_orders):
        centered=cloud-cloud.mean(axis=0);j=np.arange(size,dtype=float)
        first_sine=np.sum(centered*np.sin(2*np.pi*j/size)[:,None],axis=0)
        first_cosine=np.sum(centered*np.cos(2*np.pi*j/size)[:,None],axis=0)
        normal=np.cross(first_sine,first_cosine);norm=float(np.linalg.norm(normal))
        if norm>distance_tolerance:
            normal/=norm;z=centered@normal
            projected=centered-z[:,None]*normal
            radii=np.linalg.norm(projected,axis=1)
            even=z[::2];odd=z[1::2]
            step_angles=[]
            for index in range(size):
                left=projected[index];right=projected[(index+1)%size]
                step_angles.append(degrees(atan2(float(np.dot(normal,np.cross(left,right))),float(np.dot(left,right)))))
            crown=(min(radii)>distance_tolerance and np.max(np.abs(radii-radii.mean()))<=distance_tolerance
                and np.max(np.abs(even-even.mean()))<=distance_tolerance
                and np.max(np.abs(odd-odd.mean()))<=distance_tolerance
                and abs(even.mean()-odd.mean())>2*distance_tolerance
                and max(abs(abs(angle)-360/size) for angle in step_angles)<=angular_tolerance
                and (all(a>0 for a in step_angles) or all(a<0 for a in step_angles)))
            if crown:candidates.append(('crown',()))
    if size==8:
        for first in range(4):
            base={first,(first+1)%8,(first+4)%8,(first+5)%8}
            distances=plane_distances(cloud,tuple(sorted(base)),distance_tolerance)
            if distances is None:continue
            other=tuple(i for i in range(8) if i not in base)
            signs=distances[list(other)]
            if np.min(np.abs(signs))>distance_tolerance and (np.all(signs>0) or np.all(signs<0)):
                candidates.append(('tub',other))
    kinds={kind for kind,_ in candidates}
    if len(kinds)==1:
        kind=next(iter(kinds))
        options=tuple(sorted({tuple(atoms) for actual,atoms in candidates if actual==kind and atoms}))
        displaced=options[0] if len(options)==1 else ()
        return kind,displaced,puckering,options
    return ('ambiguous' if kinds else 'unclassified'),(),puckering,()
