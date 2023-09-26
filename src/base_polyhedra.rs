use crate::*;
use core::iter::once;
use num_traits::float::FloatConst;

/// # Base Shapes
///
/// Start shape creation methods.

impl Polyhedron {
    pub fn tetrahedron() -> Self {
        let c0 = 1.0;

        Self {
            positions: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
            ],
            face_index: vec![
                vec![2, 1, 0],
                vec![3, 2, 0],
                vec![1, 3, 0],
                vec![2, 3, 1],
            ],
            face_set_index: vec![(0..4).collect()],
            name: String::from("T"),
        }
    }

    pub fn hexahedron() -> Self {
        let c0 = 1.0;

        Self {
            positions: vec![
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![4, 5, 1, 0],
                vec![2, 6, 4, 0],
                vec![1, 3, 2, 0],
                vec![6, 2, 3, 7],
                vec![5, 4, 6, 7],
                vec![3, 1, 5, 7],
            ],
            face_set_index: vec![(0..6).collect()],
            name: String::from("C"),
        }
    }

    #[inline]
    /// Alias for [`hexahedron()`](Self::hexahedron()).
    pub fn cube() -> Self {
        Self::hexahedron()
    }

    pub fn octahedron() -> Self {
        let c0 = 0.707_106_77;

        Self {
            positions: vec![
                Point::new(0.0, 0.0, c0),
                Point::new(0.0, 0.0, -c0),
                Point::new(c0, 0.0, 0.0),
                Point::new(-c0, 0.0, 0.0),
                Point::new(0.0, c0, 0.0),
                Point::new(0.0, -c0, 0.0),
            ],
            face_index: vec![
                vec![4, 2, 0],
                vec![3, 4, 0],
                vec![5, 3, 0],
                vec![2, 5, 0],
                vec![5, 2, 1],
                vec![3, 5, 1],
                vec![4, 3, 1],
                vec![2, 4, 1],
            ],
            face_set_index: vec![(0..8).collect()],
            name: String::from("O"),
        }
    }

    pub fn dodecahedron() -> Self {
        let c0 = 0.809_017;
        let c1 = 1.309_017;

        Self {
            positions: vec![
                Point::new(0.0, 0.5, c1),
                Point::new(0.0, 0.5, -c1),
                Point::new(0.0, -0.5, c1),
                Point::new(0.0, -0.5, -c1),
                Point::new(c1, 0.0, 0.5),
                Point::new(c1, 0.0, -0.5),
                Point::new(-c1, 0.0, 0.5),
                Point::new(-c1, 0.0, -0.5),
                Point::new(0.5, c1, 0.0),
                Point::new(0.5, -c1, 0.0),
                Point::new(-0.5, c1, 0.0),
                Point::new(-0.5, -c1, 0.0),
                Point::new(c0, c0, c0),
                Point::new(c0, c0, -c0),
                Point::new(c0, -c0, c0),
                Point::new(c0, -c0, -c0),
                Point::new(-c0, c0, c0),
                Point::new(-c0, c0, -c0),
                Point::new(-c0, -c0, c0),
                Point::new(-c0, -c0, -c0),
            ],
            face_index: vec![
                vec![12, 4, 14, 2, 0],
                vec![16, 10, 8, 12, 0],
                vec![2, 18, 6, 16, 0],
                vec![17, 10, 16, 6, 7],
                vec![19, 3, 1, 17, 7],
                vec![6, 18, 11, 19, 7],
                vec![15, 3, 19, 11, 9],
                vec![14, 4, 5, 15, 9],
                vec![11, 18, 2, 14, 9],
                vec![8, 10, 17, 1, 13],
                vec![5, 4, 12, 8, 13],
                vec![1, 3, 15, 5, 13],
            ],
            face_set_index: vec![(0..12).collect()],
            name: String::from("D"),
        }
    }

    pub fn icosahedron() -> Self {
        let c0 = 0.809_017;

        Self {
            positions: vec![
                Point::new(0.5, 0.0, c0),
                Point::new(0.5, 0.0, -c0),
                Point::new(-0.5, 0.0, c0),
                Point::new(-0.5, 0.0, -c0),
                Point::new(c0, 0.5, 0.0),
                Point::new(c0, -0.5, 0.0),
                Point::new(-c0, 0.5, 0.0),
                Point::new(-c0, -0.5, 0.0),
                Point::new(0.0, c0, 0.5),
                Point::new(0.0, c0, -0.5),
                Point::new(0.0, -c0, 0.5),
                Point::new(0.0, -c0, -0.5),
            ],
            face_index: vec![
                vec![10, 2, 0],
                vec![5, 10, 0],
                vec![4, 5, 0],
                vec![8, 4, 0],
                vec![2, 8, 0],
                vec![6, 8, 2],
                vec![7, 6, 2],
                vec![10, 7, 2],
                vec![11, 7, 10],
                vec![5, 11, 10],
                vec![1, 11, 5],
                vec![4, 1, 5],
                vec![9, 1, 4],
                vec![8, 9, 4],
                vec![6, 9, 8],
                vec![3, 9, 6],
                vec![7, 3, 6],
                vec![11, 3, 7],
                vec![1, 3, 11],
                vec![9, 3, 1],
            ],
            face_set_index: vec![(0..20).collect()],
            name: String::from("I"),
        }
    }

    /// common code for prism and antiprism
    #[inline]
    fn protoprism(n: Option<usize>, anti: bool) -> Self {
        let n = n.unwrap_or(3);

        // Angles.
        let theta = f32::TAU() / n as f32;
        let twist = if anti { theta / 2.0 } else { 0.0 };
        // Half-edge.
        let h = (theta * 0.5).sin();

        let mut face_index = vec![
            (0..n).map(|i| i as VertexKey).collect::<Vec<_>>(),
            (n..2 * n).rev().map(|i| i as VertexKey).collect::<Vec<_>>(),
        ];

        // Sides.
        if anti {
            face_index.extend(
                (0..n)
                    .map(|i| {
                        vec![
                            i as VertexKey,
                            (i + n) as VertexKey,
                            ((i + 1) % n) as VertexKey,
                        ]
                    })
                    .chain((0..n).map(|i| {
                        vec![
                            (i + n) as VertexKey,
                            ((i + 1) % n + n) as VertexKey,
                            ((i + 1) % n) as VertexKey,
                        ]
                    })),
            );
        } else {
            face_index.extend((0..n).map(|i| {
                vec![
                    i as VertexKey,
                    (i + n) as VertexKey,
                    ((i + 1) % n + n) as VertexKey,
                    ((i + 1) % n) as VertexKey,
                ]
            }));
        };

        Self {
            name: format!("{}{}", if anti { "A" } else { "P" }, n),
            positions: (0..n)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() as _,
                        h,
                        (i as f32 * theta).sin() as _,
                    )
                })
                .chain((0..n).map(move |i| {
                    Point::new(
                        (twist + i as f32 * theta).cos() as _,
                        -h,
                        (twist + i as f32 * theta).sin() as _,
                    )
                }))
                .collect(),

            face_index,
            face_set_index: Vec::new(),
        }
    }

    pub fn prism(n: Option<usize>) -> Self {
        Self::protoprism(n, false)
    }

    pub fn antiprism(n: Option<usize>) -> Self {
        Self::protoprism(n, true)
    }

    pub fn pyramid(n: Option<usize>) -> Self {
        let n = n.unwrap_or(4);
        let c0 = 1.0;
        let height = c0;

        // Angle.
        let theta = f32::TAU() / n as f32;

        // bottom face
        let mut face_index =
            vec![(0..n).rev().map(|i| i as VertexKey).collect::<Vec<_>>()];

        // Sides.
        face_index.extend((0..n).map(|i| {
            vec![
                (n) as VertexKey,
                (i) as VertexKey,
                ((i + 1) % n) as VertexKey,
            ]
        }));

        Self {
            name: format!("Y{}", n),
            positions: (0..n)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() as _,
                        -c0 / 2.0,
                        (i as f32 * theta).sin() as _,
                    )
                })
                .chain(once(Point::new(0.0, -c0 / 2.0 + height, 0.0)))
                .collect(),

            face_index,
            face_set_index: Vec::new(),
        }
    }

    // Cupolas, aka Johnson Solids J3 J4 J5
    pub fn cupola(n: Option<usize>) -> Self {
        let n = n.unwrap_or(3); // sides on top polygon
        let n2 = 2 * n; // sides on bottom polygon

        // helpers
        let rt = |x| (x as f32).sqrt();
        let phi = (rt(5.) + 1.) / 2.;

        // circum radius of top polygon always p[0], of bottom polygon p[n]
        let r = [1., 0., 0., rt(3.), rt(rt(2.) + 2.), rt(phi + 2.)];

        // Height from bottom to top
        let h = [0., 0., 0., rt(2.), 1., phi - 1.];

        let theta = f32::TAU() / n as f32; // angle for top polygon
        let theta2 = f32::TAU() / n2 as f32; // angle for bottom polygon
        let twist = -theta2 / 2.0; // twist bottom for side faces

        let mut face_index = vec![
            (0..n).map(|i| i as VertexKey).collect::<Vec<_>>(), // top polygon
            (n..n + n2) // base polygon
                .rev()
                .map(|i| i as VertexKey)
                .collect::<Vec<_>>(),
        ];

        // Side faces
        face_index.extend(
            (0..n)
                .map(|i| {
                    vec![
                        i as VertexKey,
                        (n + 2 * i) as VertexKey,
                        (n + 2 * i + 1) as VertexKey,
                    ]
                })
                .chain((0..n).map(|i| {
                    vec![
                        (n + (i * 2 + 1) % n2) as VertexKey,
                        (n + (i * 2 + 2) % n2) as VertexKey,
                        ((i + 1) % n) as VertexKey,
                        (i) as VertexKey,
                    ]
                })),
        );

        Self {
            name: format!("J{}", n),
            positions: (0..n)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta).cos() * r[0] as f32,
                        h[n] / 2.0,
                        (i as f32 * theta).sin() * r[0] as f32,
                    )
                })
                .chain((0..n2).map(move |i| {
                    Point::new(
                        (twist + i as f32 * theta2).cos() * r[n] as f32,
                        -h[n] / 2.0,
                        (twist + i as f32 * theta2).sin() * r[n] as f32,
                    )
                }))
                .collect(),

            face_index,
            face_set_index: Vec::new(),
        }
    }

    // create a Johnson Solid
    // if n=unimplemented, this creates a Tetrahedron
    pub fn johnson(n: Option<usize>) -> Self {
        match n.unwrap_or(1) {
            3..=5 => Polyhedron::cupola(n),
            _ => Polyhedron::tetrahedron(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dumpobj(_p: &Polyhedron) {
        #[cfg(feature = "obj")]
        _p.write_obj(&std::path::PathBuf::from("."), false).unwrap();
    }

    // the Variance of the squared-edges of the given Polyhedron
    // useful for polyhedrons where all edges are equal
    fn edge_variance(p: &Polyhedron) -> f32 {
        let pts = p.positions();
        let eds = p.to_edges();
        let count = eds.clone().len();
        let quads = eds
            .iter()
            .map(|e| pts[e[0] as usize] - pts[e[1] as usize])
            .map(|d| d.mag_sq());
        // for q in quads.clone() { println!("{:?}",q); };
        let mean = quads.clone().sum::<f32>() / count as f32;
        quads.map(|d| (d - mean) * (d - mean)).sum::<f32>()
            / (count as f32 - 1.0f32)
    }

    #[test]
    fn test_pyramid() {
        let p = Polyhedron::pyramid(Some(4));
        dumpobj(&p);
        assert!(p.faces().len() == 5);
        assert!(p.positions_len() == 5);
        assert!(p.to_edges().len() == 8);
    }

    #[test]
    fn test_cupola() {
        for n in [3, 4, 5] {
            let p = Polyhedron::cupola(Some(n));
            dumpobj(&p);
            let f = p.faces().len();
            let v = p.positions_len();
            let e = p.to_edges().len();
            assert!(f == n * 2 + 2);
            assert!(v == n + n * 2);
            assert!(e == n + n * 2 + n * 2);
            assert!(f + v - e == 2); // Euler's Formula
            assert!(edge_variance(&p) < 0.001);
        }
    }

    #[test]
    fn test_polyhedra() {
        for p in [
            Polyhedron::tetrahedron(),
            Polyhedron::icosahedron(),
            Polyhedron::prism(Some(4)),
            Polyhedron::prism(Some(7)),
            Polyhedron::antiprism(Some(3)),
            Polyhedron::antiprism(Some(8)),
            Polyhedron::pyramid(Some(4)),
            Polyhedron::pyramid(Some(9)),
            Polyhedron::cupola(Some(3)),
        ] {
            dumpobj(&p);
            let f = p.faces().len();
            let v = p.positions_len();
            let e = p.to_edges().len();
            assert!(f + v - e == 2); // Euler's Formula
        }
    }
}
