use kiss3d::{
    camera::ArcBall,
    context::Context,
    light::Light,
    nalgebra as na,
    resource::{Material, Mesh},
    window::Window,
};
use na::{Point3, Vector3};
use polyhedron_ops::*;
use std::{cell::RefCell, env, error::Error, io, io::Write, rc::Rc};

use itertools::Itertools;

mod material;

fn into_mesh(polyhedron: Polyhedron) -> Mesh {
    let (face_index, points, normals) = polyhedron.to_triangle_mesh_buffers();

    Mesh::new(
        // Duplicate points per face so we can
        // match the normals per face.
        points
            .iter()
            .map(|p| na::Point3::<f32>::new(p.x, p.y, p.z))
            .collect::<Vec<_>>(),
        face_index
            .iter()
            .tuples::<(_, _, _)>()
            .map(|i| na::Point3::<u16>::new(*i.0 as _, *i.1 as _, *i.2 as _))
            .collect::<Vec<_>>(),
        Some(
            normals
                .iter()
                .map(|n| na::Vector3::new(n.x, n.y, n.z))
                .collect::<Vec<_>>(),
        ),
        None,
        false,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let mut poly = Polyhedron::icosidodecahedron();
    //        Polyhedron::icosahedron();
    //Polyhedron::rhombicosidodecahedron();
    poly.normalize();
    //	poly.reverse();
    let distance = 2.0f32;
    let eye = Point3::new(distance, distance, distance);
    let at = Point3::origin();
    let mut arc_ball = ArcBall::new(eye, at);

    let mut window = Window::new("Polyhedron Operations");
    //        let ctxt = Context::get();
    //	ctxt.enable(Context::BLEND);
    //	let mut s = window.add_sphere(1.0);
    //    s.set_material(material.clone());

    let mesh = Rc::new(RefCell::new(into_mesh(poly.clone())));
    let material = Rc::new(RefCell::new(
        Box::new(material::NormalMaterial::new()) as Box<dyn Material + 'static>
    ));

    let mut c = window.add_mesh(mesh.clone(), Vector3::new(1.0, 1.0, 1.0));
    c.set_material(material);
    c.enable_backface_culling(false);
    c.set_points_size(10.);

    window.set_light(Light::StickToCamera);
    window.set_framerate_limit(Some(60));

    let path = dirs::home_dir().unwrap();

    println!("{}", poly.name());
    //    io::stdout().flush().unwrap();

    while !window.should_close() {
        // rotate the arc-ball camera.
        let curr_yaw = arc_ball.yaw();
        arc_ball.set_yaw(curr_yaw + 0.001);

        // x y z axes
        window.draw_line(
            &Point3::new(-1.618, 0.0, 0.0),
            &Point3::new(1.618, 0.0, 0.0),
            &Point3::new(1.618, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::new(0.0, -1.618, 0.0),
            &Point3::new(0.0, 1.618, 0.0),
            &Point3::new(0.0, 1.618, 0.0),
        );
        window.draw_line(
            &Point3::new(0.0, 0.0, -1.618),
            &Point3::new(0.0, 0.0, 1.618),
            &Point3::new(0.0, 0.0, 1.618),
        );


        // polyhedron edges

        //window.set_line_width(1.0);

        /*for e in poly.to_edges() {
            let p1 = poly.positions()[e[0] as usize];
            let p2 = poly.positions()[e[1] as usize];
            let kp1 = Point3::new(p1.x, p1.y, p1.z);
            let kp2 = Point3::new(p2.x, p2.y, p2.z);
            let color = Point3::new(1.0, 1.0, 1.0);
            window.draw_line(&kp1, &kp2, &color);
            //        println!("{:?} {:?} {:?}",kp1, kp2,color );
        }*/

	// Polyhedron
        window.render_with_camera(&mut arc_ball);

    }

    Ok(())
}
