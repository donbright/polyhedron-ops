use crate::*;
use core::iter::once;
use num_traits::float::FloatConst;
//use itertools::Itertools;
use itertools::iproduct;
use chull::ConvexHullWrapper;

/// # Base Shapes
///
/// Start shape creation methods.

// take a string "x" and return x a floating point number
// special symbols understood:
//    Φ golden ratio 
//	  Φ′ golden ratio conjugate
//    Φ⁻¹ inverse golden ratio
//    √2 square root of 2
//    etc etc

fn floatify(s:&str) -> f32 {
    let phi = (1. + 5.0f32.sqrt()) / 2.;
    let phiprime = (1. - 5.0f32.sqrt()) / 2.;
    println!(":{}", s);
	let s = s.replace("φ","Φ");
    let f = s.replace("√(Φ′+2)", &format!("{}",(phiprime+2.).sqrt()))
		.replace("√(Φ+2)", &format!("{}",(phi+2.).sqrt()))
		.replace("Φ-1", &format!("{}", phi-1.))
		.replace("Φ²", &format!("{}", phi*phi))
        .replace("Φ⁻¹", &format!("{}", 1./phi))
		.replace("Φ/2", &format!("{}", phi/2.))
		.replace("Φ′",&format!("{}",phiprime))
		.replace("1+√2",&format!("{}",1.+2.0f32.sqrt()))
		.replace("√3", &format!("{}", 3.0f32.sqrt()))
		.replace("√2", &format!("{}", 2.0f32.sqrt()))
        .replace("Φ", &format!("{}", phi))
        .replace("--", "")
        .replace("_", ""); // mimic rust float literal syntax
	println!("{:?}",f);
        f.parse::<f32>().unwrap()
}

// given a string like = "±2,0,0" expand the plusminus and create
// vectors of floating point, [+2.,0.,0.][-2.,2.,0.]
// for "±2,0,±1" generate [-2,0,1][2,0,1][-2,0,-1][2,0,-1], etc.
// the numbers after ± are processed by floatify() so it can
// understand Φ as the golden ratio, and other symbols.
fn seed_points(s: &str) -> impl Iterator<Item = Point> {
	let mut v:Vec<Vec<f32>> = Vec::new();
	for n in s.split(",") { 
		match n.chars().nth(0).unwrap() {
			 '±'=> {
				let f = floatify( n.split("±").nth(1).unwrap() );
				v.push( vec![1. * f, -1. * f] );
			 },
			 _=> v.push( vec![ floatify(n) ])
		}
	};
	iproduct!(v[0].clone(),v[1].clone(),v[2].clone()).map(|v|Point::new(v.0,v.1,v.2))
}

#[cfg(test)]
#[test]
fn test_seed_points() {
	for p in seed_points("0,0,±1") { println!("{:?}",p); }
        itertools::assert_equal( seed_points("0,0,±1")
           , vec![Point::new(0., 0., 1.), Point::new(0., 0., -1.)] );

	itertools::assert_equal( 
        seed_points("±2,0,±1")
            , vec![
                Point::new(2., 0., 1.),
                Point::new(2., 0., -1.),
                Point::new(-2., 0., 1.),
                Point::new(-2., 0., -1.)
            ]);
}

// make rotations(n) iterator adapter
trait RotationsValue {
	fn rotations_value(self)->Point;
}
impl RotationsValue for Point { 
	fn rotations_value(self)-> Point {
		self
	}
}

impl RotationsValue for &Point { 
	fn rotations_value(self)-> Point {
		*self
	}
}

struct Rotations<I> {
	inner: I,
	last_point: Option<Point>,
	start_count: u8,
	count: u8,
}

impl<I, T> Iterator for Rotations<I> 
where 
I:Iterator<Item = T>,
T:RotationsValue,
{
	type Item = Point;
	fn next(&mut self)->Option<Self::Item> {
		println!("next. sc{} c{}",self.start_count,self.count);
		if self.count == self.start_count {
			println!("m1 next. sc{} c{}",self.start_count,self.count);
			self.count -= 1;
			self.last_point = self.inner.next().map(|x| x.rotations_value());
		} else  {	self.count -= 1;
			let p = self.last_point.unwrap();
			if self.count == 0{ self.count = self.start_count;}
			self.last_point = Some(Point::new( p.z, p.x, p.y ))
		}
		self.last_point
	}
}

trait RotationsAdapter: Iterator {
    fn rotations(self,count:u8) -> Rotations<Self>
    where
        Self: Sized,
    {
        Rotations { inner: self, last_point: None, start_count: count, count: count }
    }
}

impl<I,T> RotationsAdapter for I 
where 
I: Iterator<Item=T>,
 T: RotationsValue ,
{}

#[cfg(test)]
#[test]
fn test_rotations() {
    itertools::assert_equal(
        vec![Point::new(1., 2., 3.)].iter().rotations(3)
            , vec![
                Point::new(1., 2., 3.),
                Point::new(3., 1., 2.),
                Point::new(2., 3., 1.)
            ].into_iter()
    );
    itertools::assert_equal(
        vec![Point::new(1., 2., 3.)].iter().rotations(2)
            , vec![
                Point::new(1., 2., 3.),
                Point::new(3., 1., 2.),
            ].into_iter()
    );
    itertools::assert_equal(
        vec![Point::new(1., 2., 3.), Point::new(1., 2., -3.)].iter().rotations(3)
            , vec![
                Point::new(1., 2., 3.),
                Point::new(3., 1., 2.),
                Point::new(2., 3., 1.),
                Point::new(1., 2., -3.),
                Point::new(-3., 1., 2.),
                Point::new(2., -3., 1.)
            ].into_iter()
    );
}

impl Polyhedron {
    pub fn tetrahedron() -> Self {
        Self {
            positions: 
				seed_points("1,1,1").chain(
                seed_points("1,-1,-1").rotations(3)).collect()
            ,
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
            positions: seed_points("±1,±1,±1").collect(),
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
            positions: seed_points("0,0,±0.707_106_77").rotations(3).collect(),
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
		let pts = seed_points("0,1,±Φ²").rotations(3).chain(
				seed_points("±Φ,±Φ,±Φ")).collect();
        Self {
            positions: pts,
            face_index: convex_hull(pts),
            face_set_index: vec![(0..12).collect()],
            name: String::from("D"),
        }
    }

    pub fn icosahedron() -> Self {
        Self {
            positions: seed_points("±1,0,±Φ").rotations(3).collect(),
            face_index: vec![
                vec![6,5,0], vec![10,5,6], 
                vec![1,8,2], vec![1,3,8],vec![4,8,9], vec![9,10,4],
                vec![10,9,11], vec![10,11,5],vec![11,7,5], vec![7,0,5],
                vec![1,7,3],  vec![1,0,7],vec![1,2,0], vec![0, 2,6],
                vec![6,2,4],  vec![2,8,4],vec![10,6,4], vec![3,7,11],
                vec![9,3,11], vec![9,8,3],
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

    pub fn triangular_cupola() -> Self {
        Self {
            positions: 
				seed_points("±2,0,0").chain(
                seed_points("±1,±√3,0")).chain(
				seed_points("±1,√3,√3")).chain(
				seed_points("1,-2/√3,√3")).collect()
            ,
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

    pub fn square_cupola() -> Self {
        Self {
            positions: 
				seed_points("±1,±1+√2,0").chain(
				seed_points("±1+√2,±1,0")).chain(
                seed_points("±1,±1,√2")).collect()
            ,
            face_index: vec![
                vec![2,6,7,3,1,5,4,0],
                vec![10,8,9,11],
                vec![1,3,11,9],   
                vec![2,0,8,10],   
                vec![4,5,9,8],    
                vec![6,10,11,7],  
                vec![0,4,8],       
                vec![5,1,9],       
                vec![3,7,11],      
                vec![6,2,10],       
            ],
            face_set_index: vec![(0..4).collect()],
            name: String::from("J4"),
        }
    }

	pub fn rhombicosidodecahedron() -> Self {
		let points = seed_points("±1, ±1, ±φ^3").chain(
    		seed_points("±φ2, ±φ, ±2")).chain( 
			seed_points("±(2+φ), 0, ±φ^2")).collect();
        Self {
            positions: points,
            face_index: vec![vec![1,2,3]], //convex_hull(points),
            face_set_index: vec![(0..1).collect()],
            name: String::from("eD"),
		}
	}

    pub fn pentagonal_cupola() -> Self {
		//let p = Self::rhombicosidodecahedron();
		let p = Self::icosahedron();
		let points = p.positions().iter().cloned().collect::<Points>();
//.filter(|p|p.z>0.)
		let mut v = vec![];
		for q in &points {
			v.push(vec![q.x,q.y,q.z]);
		};
		println!("v:{:?}",v);
		let hull = ConvexHullWrapper::try_new(&v, None).unwrap();
		println!("hull:{:?}",hull.vertices_indices().1);
		let m = (hull.vertices_indices().1).into_iter().collect::<Vec<usize>>();
		let m2:Vec<u32> = m.into_iter().map(|x| x as u32).collect();
		let fi: Vec<Vec<u32>> =  m2.chunks(3)
                                 .map(|chunk| chunk.to_vec()).collect();
		println!("fi:{:?}",fi);
        Self {
            positions: points,
            face_index: fi,//vec![vec![0,1,2]],//convex_hull(points),
            face_set_index: vec![(0..1).collect()],
            name: String::from("J5"),
        }
    }

    // Rotunda, Johnson Solid J6
    pub fn rotunda() -> Self {
        //   let s = "±2,0,0";
        //    let s = "±Φ,±Φ⁻¹,±1" ;

        let n = [5, 5, 10]; // sides on top, middle, and bottom polygon

        // helpers
        let rt = |x| (x as f32).sqrt();
        let mktheta = |n| f32::TAU() / (n as f32);
        let phi = (rt(5.) + 1.) / 2.;

        // circum radius of top, mid, and bottom polygon
        let r = [1., phi, rt(phi + 2.)];

        // height of top-to-mid, and mid-to-bottom
        let h = [0., phi - 1., (phi + 1.) / 4.];
        let htot = h[1] + h[2];

        // angles for top, mid, bottom polygon
        let theta = [5, 5, 10].map(|n| mktheta(n));
        // twist middle polygon
        let twist = [0., theta[1] / 2., 0.];

        // top, mid, and bottom polygons
        let mut face_index = vec![
            (0..5).map(|i| i as VertexKey).collect::<Vec<_>>(),
            (0..5).rev().map(|i| i as VertexKey).collect::<Vec<_>>(),
            (0..10).map(|i| i as VertexKey).collect::<Vec<_>>(),
        ];

        // Side faces
        face_index.extend(
            (0..5)
                .map(|i| {
                    vec![
                        i as VertexKey,
                        (i+1) as VertexKey,
                        (5+i) as VertexKey
                    ]
                })
                .chain((0..5).map(|i| {
                    vec![
						(5+i) as VertexKey,
						(5+5+i) as VertexKey,
						(5+5+i+1) as VertexKey
                    ]
                }))
//,
/*                .chain((0..10).map(|i| {
                    vec![
                        (n + (i * 2 + 1) % n2) as VertexKey,
                        (n + (i * 2 + 2) % n2) as VertexKey,
                        ((i + 1) % n) as VertexKey,
                        (i) as VertexKey,
                    ]
                })),*/
        );

        Self {
            name: format!("J6"),
            positions: (0..5)
                .map(move |i| {
                    Point::new(
                        (i as f32 * theta[0]).cos() * r[0] as f32,
                        htot / 2.,
                        (i as f32 * theta[0]).sin() * r[0] as f32,
                    )
                })
                .chain((0..5).map(move |i| {
                    Point::new(
                        (twist[1] + i as f32 * theta[1]).cos() * r[1] as f32,
                        -htot / 2.0 + h[2],
                        (twist[1] + i as f32 * theta[1]).sin() * r[1] as f32,
                    )
                }))
                .chain((0..10).map(move |i| {
                    Point::new(
                        (i as f32 * theta[2]).cos() * r[2] as f32,
                        -htot / 2.0,
                        (i as f32 * theta[2]).sin() * r[2] as f32,
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
            1..=2 => Polyhedron::pyramid(Some(n.unwrap_or(1) + 3)),
            3 => Polyhedron::triangular_cupola(),
            4 => Polyhedron::square_cupola(),
            5 => Polyhedron::pentagonal_cupola(),
            6 => Polyhedron::rotunda(),
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
       // let p = Polyhedron::square_cupola();
        let p = Polyhedron::pentagonal_cupola();
        dumpobj(&p);
//        for n in [3, 4, 5] {
//            let p = Polyhedron::cupola(Some(n));
//            dumpobj(&p);
			let n = 5;
            let f = p.faces().len();
            let v = p.positions_len();
            let e = p.to_edges().len();
            assert!(f == n * 2 + 2);
            assert!(v == n + n * 2);
            assert!(e == n + n * 2 + n * 2);
            assert!(f + v - e == 2); // Euler's Formula
            assert!(edge_variance(&p) < 0.001);
//        }
    }

    #[test]
    fn test_rotunda() {
        /*            let p = Polyhedron::rotunda();
                  dumpobj(&p);
                  let f = p.faces().len();
                  let v = p.positions_len();
                  let e = p.to_edges().len();
                  assert!(f == 17);
                  assert!(v == 5+5+10);
                  assert!(e == 5*5+5+5);
                  assert!(f + v - e == 2); // Euler's Formula
                  assert!(edge_variance(&p) < 0.001);
        */
    }

    #[test]
    fn test_variance() {
        assert!(edge_variance(&Polyhedron::hexahedron()) < 0.001);
        assert!(edge_variance(&Polyhedron::tetrahedron()) < 0.001);
        assert!(edge_variance(&Polyhedron::icosahedron()) < 0.001);
    }

    #[test]
    fn test_polyhedra() {
        for p in [
            Polyhedron::hexahedron(),
            Polyhedron::dodecahedron(),
            Polyhedron::tetrahedron(),
            Polyhedron::icosahedron(),
            Polyhedron::prism(Some(4)),
            Polyhedron::prism(Some(7)),
            Polyhedron::antiprism(Some(3)),
            Polyhedron::antiprism(Some(8)),
            Polyhedron::pyramid(Some(4)),
            Polyhedron::pyramid(Some(9)),
            Polyhedron::johnson(Some(3)),
            Polyhedron::johnson(Some(5)),
        ] {
            dumpobj(&p);
            let f = p.faces().len();
            let v = p.positions_len();
            let e = p.to_edges().len();
            assert!(f + v - e == 2); // Euler's Formula
        }
    }
}


fn square_pyramid()
fn pentagonal_pyramid()
fn triangular_cupola()
fn square_cupola()
fn pentagonal_cupola()
fn pentagonal_rotunda()
fn elongated_triangular_pyramid()
fn elongated_square_pyramid()
fn elongated_pentagonal_pyramid()
fn gyroelongated_square_pyramid()
fn gyroelongated_pentagonal_pyramid()
fn triangular_dipyramid()
fn pentagonal_dipyramid()
fn elongated_triangular_dipyramid()
fn elongated_square_dipyramid()
fn elongated_pentagonal_dipyramid()
fn gyroelongated_square_dipyramid()
fn elongated_triangular_cupola()
fn elongated_square_cupola()
fn elongated_pentagonal_cupola()
fn elongated_pentagonal_rotunda()
fn gyroelongated_triangular_cupola()
fn gyroelongated_square_cupola()
fn gyroelongated_pentagonal_cupola()
fn gyroelongated_pentagonal_rotunda()
fn gyrobifastigium()
fn triangular_orthobicupola()
fn square_orthobicupola()
fn square_gyrobicupola()
fn pentagonal_orthobicupola()
fn pentagonal_gyrobicupola()
fn pentagonal_orthocupolarotunda()
fn pentagonal_gyrocupolarotunda()
fn pentagonal_orthobirotunda()
fn elongated_triangular_orthobicupola()
fn elongated_triangular_gyrobicupola()
fn elongated_square_gyrobicupola()
fn elongated_pentagonal_orthobicupola()
fn elongated_pentagonal_gyrobicupola()
fn elongated_pentagonal_orthocupolarotunda()
fn elongated_pentagonal_gyrocupolarotunda()
fn elongated_pentagonal_orthobirotunda()
fn elongated_pentagonal_gyrobirotunda()
fn gyroelongated_triangular_bicupola()
fn gyroelongated_square_bicupola()
fn gyroelongated_pentagonal_bicupola()
fn gyroelongated_pentag()
fn gyroelongated_pentagonal_birotunda()
fn augmented_triangular_prism()
fn biaugmented_triangular_prism()
fn triaugmented_triangular_prism()
fn augmented_pentagonal_prism()
fn biaugmented_pentagonal_prism()
fn augmented_hexagonal_prism()
fn parabiaugmented_hexagonal_prism()
fn metabiaugmented_hexagonal_prism()
fn triaugmented_hexagonal_prism()
fn augmented_dodecahedron()
fn parabiaugmented_dodecahedron()
fn metabiaugmented_dodecahedron()
fn triaugmented_dodecahedron()
fn metabidiminished_icosahedron()
fn tridiminished_icosahedron()
fn augmented_tridiminished_icosahedron()
fn augmented_truncated_tetrahedron()
fn augmented_truncated_cube()
fn biaugmented_truncated_cube()
fn augmented_truncated_dodecahedron()
fn parabiaugmented_truncated()
fn metabiaugmented_truncated()
fn triaugmented_truncated_dodecahedron()
fn gyrate_rhombicosidodecahedron()
fn parabigyrate_rhombicosidodecahedron()
fn metabigyrate_rhombicosidodecahedron()
fn trigyrate_rhombicosidodecahedron()
fn diminished_rhombicosidodecahedron()
fn paragyrate_diminished()
fn metagyrate_diminished()
fn bigyrate_diminished()
fn parabidiminished_rhombicosidodecahedron()
fn metabidiminished_rhombicosidodecahedron()
fn gyrate_bidiminished()
fn tridiminished_rhombicosidodecahedron()
fn snub_disphenoid()
fn snub_square_antiprism()
fn sphenocorona()
fn augmented_sphenocorona()
fn sphenomegacorona()
fn hebesphenomegacorona()
fn disphenocingulum()
fn bilunabirotunda()
fn triangular_hebesphenorotunda()
fn ()
