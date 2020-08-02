use cgmath::prelude::*;
//use itertools::Itertools;
use nsi;
use std::iter::{once, Iterator};

type Float = f32;
type Index = u32;
type Face = Vec<Index>;
type FaceIndex = Vec<Face>;
type EdgeIndex = Vec<Edge>;
type Edge = (Index, Index);

type FlatVertices = Vec<Float>;
type Point = cgmath::Point3<Float>;
type Vector = cgmath::Vector3<Float>;
type Vertices = Vec<Point>;
type VerticesRef<'a> = Vec<&'a Point>;

#[derive(Clone, Debug)]
struct Mesh {
    vertices: Vertices,
    //face_arity: Vec<index>,
    face_index: FaceIndex,
}

/// Returns a [`FaceIndex`] of faces
/// containing `vertex_number`.
#[inline]
fn vertex_faces(vertex_number: Index, face_index: &FaceIndex) -> FaceIndex {
    face_index
        .iter()
        .filter(|face| face.contains(&vertex_number))
        .cloned()
        .collect()
}

/*
function ordered_face_edges(f) =
    // edges are ordered anticlockwise
    [for (j=[0:len(f)-1])
        [f[j],f[(j+1)%len(f)]]
    ];
*/
/// Returns a [`Vec`] of anticlockwise
/// ordered edges.
/*fn ordered_face_edges(f: &Face) -> EdgeIndex {
    f.iter()
        .cycle()
        .tuple_windows::<(_, _)>()
        .take(f.len())
        .collect()
}*/

#[inline]
fn ordered_face_edges(f: &Face) -> EdgeIndex {
    let mut result = EdgeIndex::new();
    for v in 0..f.len() {
        result.push((f[v], f[(v + 1) % f.len()]));
    }
    result
}

/*
function face_with_edge(edge,faces) =
    flatten(
    [for (f = faces)
        if (vcontains(edge,ordered_face_edges(f)))
        f
    ]);
*/
#[inline]
fn face_with_edge(edge: Edge, faces: &FaceIndex) -> Face {
    let result = faces
        .iter()
        .filter(|face| ordered_face_edges(face).contains(&edge))
        .flatten()
        .cloned()
        .collect();
    result
}

/*
function ordered_vertex_faces(v,vfaces,cface=[],k=0) =
    k==0
    ? let (nface=vfaces[0])
        concat([nface],ordered_vertex_faces(v,vfaces,nface,k+1))
    : k < len(vfaces)
        ?  let(i = index_of(v,cface))
           let(j= (i-1+len(cface))%len(cface))
           let(edge=[v,cface[j]])
           let(nface=face_with_edge(edge,vfaces))
              concat([nface],ordered_vertex_faces(v,vfaces,nface,k+1 ))
        : []
*/
#[inline]
fn index_of<T: PartialEq>(element: &T, list: &Vec<T>) -> Option<usize> {
    list.iter().position(|e| *e == *element)
}

/// Used internally by ordered_vertex_faces.
#[inline]
fn ordered_vertex_faces_recurse(
    v: Index,
    face_index: &FaceIndex,
    cface: &Face,
    k: Index,
) -> FaceIndex {
    if (k as usize) < face_index.len() {
        let i = index_of(&v, &cface).unwrap() as i32;
        let j = ((i - 1 + cface.len() as i32) % cface.len() as i32) as usize;
        let edge = (v, cface[j]);
        let mut nfaces = vec![face_with_edge(edge, face_index)];
        nfaces.extend(ordered_vertex_faces_recurse(
            v,
            face_index,
            &nfaces[0],
            k + 1,
        ));
        nfaces
    } else {
        FaceIndex::new()
    }
}

#[inline]
fn ordered_vertex_faces(vertex_number: Index, face_index: &FaceIndex) -> FaceIndex {
    let mut result = vec![face_index[0].clone()];
    result.extend(ordered_vertex_faces_recurse(
        vertex_number,
        face_index,
        &face_index[0],
        1,
    ));

    result
}

#[inline]
fn as_points<'a>(f: &'a Face, vertices: &'a Vertices) -> VerticesRef<'a> {
    f.iter().map(|index| &vertices[*index as usize]).collect()
}

fn centroid_ref<'a>(vertices: &'a VerticesRef<'a>) -> Point {
    let total_displacement = vertices
        .into_iter()
        .fold(Vector::new(0., 0., 0.), |acc, p| acc + (**p).to_vec());

    Point::from_vec(total_displacement / vertices.len() as f32)
}

#[inline]
fn orthogonal(v0: &Point, v1: &Point, v2: &Point) -> Vector {
    (v1 - v0).cross(v2 - v1)
}

/// Computes the normal of a face.
/// Assumes the face is planar.
#[inline]
fn face_normal(vertices: &VerticesRef) -> Vector {
    // FIXME iterate over all points, calculate normal everywhere and average
    -orthogonal(&vertices[0], &vertices[1], &vertices[2]).normalize()
}

fn vertex_ids(entries: Vec<(&Face, Point)>, offset: usize) -> Vec<(&Face, usize)> {
    entries
        .iter()
        .enumerate()
        .map(|i| (i.1.0, i.0 + offset))
        .collect()
}

impl Mesh {
    //[ for (f=faces) if(v!=[] && search(v,f)) f ];

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn p_vertices_to_faces(&mut self) -> FaceIndex {
        let mut fi = FaceIndex::new();
        for vertex_number in 0..self.num_vertices() as u32 {
            // each old vertex creates a new face, with
            let vertex_faces = vertex_faces(vertex_number, &self.face_index);
            // vertex faces in left-hand order

            let mut new_face = Face::new();
            for of in ordered_vertex_faces(vertex_number as u32, &vertex_faces) {
                new_face.push(index_of(&of, &self.face_index).unwrap() as Index)
            }
            fi.push(new_face)
        }
        fi
    }

    /*
    function dual(obj) =
    poly(name=str("d",p_name(obj)),
         vertices =
              [for (f = p_faces(obj))
               let(fp=as_points(f,p_vertices(obj)))
                 centroid(fp)
              ],
         faces= p_vertices_to_faces(obj)
        )
    ;  // end dual
    */

    fn dual(&mut self) {
        self.vertices = self
            .face_index
            .iter()
            .map(|face| centroid_ref(&as_points(face, &self.vertices)))
            .collect();
        self.face_index = self.p_vertices_to_faces();
    }

    /// kis each n-face is divided into n triangles which extend to the face centroid
    /// existimg vertices retained
    fn kis(&mut self, height: Float, f_n: usize) {
        let pf = &self.face_index;
        let pv = &self.vertices;

        let newv: Vec<(&Face, Point)> = pf
            .iter()
            .filter(|face| f_n == face.len()) // new centroid vertices
            .map(|face| {
                let fp = as_points(face, pv);
                (face, centroid_ref(&fp) + face_normal(&fp) * height)
            })
            .collect();

        let newids = vertex_ids(newv, pv.len());

        /*let(newf=
            flatten(
              [for (face=pf)
                 selected_face(face,fn)
              //replace face with triangles
                   ? let(centroid=vertex(face,newids))
                     [for (j=[0:len(face)-1])
                      let(a=face[j],
                          b=face[(j+1)%len(face)])
                       [a,b,centroid]
                     ]
                   : [face]                              // original face
              ])
             )

        poly(name=str("k",p_name(obj)),
            vertices= concat(pv, vertex_values(newv)) ,
            faces=newf
        )*/
    } // end kis

    pub fn to_nsi(&self, ctx: nsi::Context, name: &str) {
        // Create a new mesh node and call it 'dodecahedron'.
        ctx.create(name, nsi::NodeType::Mesh, &[]);

        // Connect the 'dodecahedron' node to the scene's root.
        ctx.connect(name, "", nsi::NodeType::Root, "objects", &[]);

        /*
        let positions: FlatVertices = self
            .vertices
            .into_iter()
            .flat_map(|p3| once(p3.x).chain(once(p3.y)).chain(once(p3.z)))
            .collect();
        */

        let positions = unsafe {
            std::slice::from_raw_parts(
                self.vertices.as_ptr().cast::<Float>(),
                3 * self.vertices.len(),
            )
        };

        let face_arity = self
            .face_index
            .iter()
            .map(|face| face.len() as u32)
            .collect::<Vec<u32>>();

        let face_index = self.face_index.concat();

        ctx.set_attribute(
            name,
            &[
                nsi::points!("P", positions),
                nsi::unsigneds!("P.indices", &face_index),
                // 5 vertices per each face.
                nsi::unsigneds!("nvertices", &face_arity),
                // Render this as a subdivison surface.
                // nsi::string!("subdivision.scheme", "catmull-clark"),
                // Crease each of our 30 edges a bit.
                //nsi::unsigneds!("subdivision.creasevertices", &face_index),
                //nsi::floats!("subdivision.creasesharpness", &[10.; 30]),
            ],
        );
    }
}

/*
impl<T> From<(Vec<T>, Vec<Vec<Index>>)> for Mesh<T> {
    fn from(mesh_tuple: (Vec<T>, Vec<Vec<Index>>)) -> Mesh<T> {}
}*/

#[cfg(test)]
mod tests {

    fn tetrahedron_to_terahedron() {
        let c0 = 0.353553390593273762200422181052;
        // Tetrahedron

        // 3 sided faces = 4
        // Tetrahedron
        let mut tetrahedron = crate::Mesh {
            vertices: vec![
                cgmath::Point3::new(c0, -c0, c0),
                cgmath::Point3::new(c0, c0, -c0),
                cgmath::Point3::new(-c0, c0, c0),
                cgmath::Point3::new(-c0, -c0, -c0),
            ],
            face_index: vec![vec![2, 1, 0], vec![3, 0, 1], vec![0, 3, 2], vec![1, 2, 3]],
        };

        println!("{:?}", tetrahedron);

        tetrahedron.dual();

        println!("{:?}", tetrahedron);
    }

    #[test]
    fn cube_to_octahedron() {
        let mut cube = crate::Mesh {
            vertices: vec![
                cgmath::Point3::new(0.5, 0.5, 0.5),
                cgmath::Point3::new(0.5, 0.5, -0.5),
                cgmath::Point3::new(0.5, -0.5, 0.5),
                cgmath::Point3::new(0.5, -0.5, -0.5),
                cgmath::Point3::new(-0.5, 0.5, 0.5),
                cgmath::Point3::new(-0.5, 0.5, -0.5),
                cgmath::Point3::new(-0.5, -0.5, 0.5),
                cgmath::Point3::new(-0.5, -0.5, -0.5),
            ],
            face_index: vec![
                vec![4, 5, 1, 0],
                vec![2, 6, 4, 0],
                vec![1, 3, 2, 0],
                vec![6, 2, 3, 7],
                vec![5, 4, 6, 7],
                vec![3, 1, 5, 7],
            ],
        };

        println!("{:?}", cube);

        cube.dual();

        println!("{:?}", cube);
    }
}
