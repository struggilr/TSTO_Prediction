"""Read every 2-D volume section; split QUAD_4 with positive triangles."""
import numpy as np
import cgns_tsto_io as legacy

def parse_cgns(path):
    data=legacy.parse_cgns(path)
    base,zone=legacy.load_tree(path)
    points=np.column_stack([data['x'],data['y']]);triangles=[];counts={}
    for section in zone[2]:
        if section[3]!='Elements_t':continue
        count={}
        for kind,ids in legacy.mixed_elements(section):
            count[str(kind)]=count.get(str(kind),0)+1
            if kind==3:continue
            if kind==5:
                triangles.append(ids)
                continue
            elif kind==7:candidates=[ids[[[0,1,2],[0,2,3]]],ids[[[0,1,3],[1,2,3]]]]
            else:raise ValueError(f'Unsupported volume element {kind}')
            accepted=None
            for candidate in candidates:
                p=points[candidate]
                a=p[:,1]-p[:,0];b=p[:,2]-p[:,0];areas=a[:,0]*b[:,1]-a[:,1]*b[:,0]
                if np.all(areas<0):candidate=candidate[:,[0,2,1]];areas=-areas
                if np.all(areas>0):accepted=candidate;break
            if accepted is None:raise ValueError('Degenerate or invalid source volume cell')
            triangles.extend(accepted)
        counts[section[0]]=count
    tri=np.asarray(triangles,dtype=np.int64)
    a=points[tri[:,1]]-points[tri[:,0]];b=points[tri[:,2]]-points[tri[:,0]]
    signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.any(signed==0):raise ValueError('Degenerate source triangle')
    tri[signed<0]=tri[signed<0][:,[0,2,1]]
    data['triangles']=tri
    data['source_section_counts']=counts
    data['unreferenced_nodes']=len(data['x'])-len(np.unique(data['triangles']))
    return data
