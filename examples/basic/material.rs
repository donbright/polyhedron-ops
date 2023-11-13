use kiss3d::{
    camera::Camera,
    context::Context,
    light::Light,
    nalgebra as na,
    resource::{Effect, Material, Mesh, ShaderAttribute, ShaderUniform},
    scene::ObjectData,
};
use na::{Isometry3, Matrix3, Matrix4, Point3, Vector3};

// A material that draws normals
pub struct NormalMaterial {
    shader: Effect,
    position: ShaderAttribute<Point3<f32>>,
    normal: ShaderAttribute<Vector3<f32>>,
    view: ShaderUniform<Matrix4<f32>>,
    proj: ShaderUniform<Matrix4<f32>>,
    transform: ShaderUniform<Matrix4<f32>>,
    scale: ShaderUniform<Matrix3<f32>>,
}

impl NormalMaterial {
    pub fn new() -> NormalMaterial {
        let mut shader =
            Effect::new_from_str(NORMAL_VERTEX_SRC, NORMAL_FRAGMENT_SRC);

        //////////////

        //        shader.use_program();
        let ctxt = Context::get();
        shader.use_program();

        // this stuff enables alpha (transparency)
        let _ = ctxt.polygon_mode(Context::FRONT_AND_BACK, Context::FILL);
        ctxt.enable(Context::BLEND);
        ctxt.blend_func_separate(
            Context::SRC_ALPHA,
            Context::ONE_MINUS_SRC_ALPHA,
            Context::ONE,
            Context::ONE_MINUS_SRC_ALPHA,
        );
        ctxt.disable(Context::DEPTH_TEST);

        ///////////////
        NormalMaterial {
            position: shader.get_attrib("position").unwrap(),
            normal: shader.get_attrib("normal").unwrap(),
            transform: shader.get_uniform("transform").unwrap(),
            scale: shader.get_uniform("scale").unwrap(),
            view: shader.get_uniform("view").unwrap(),
            proj: shader.get_uniform("proj").unwrap(),
            shader,
        }
    }
}

impl Material for NormalMaterial {
    fn render(
        &mut self,
        pass: usize,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        camera: &mut dyn Camera,
        _: &Light,
        _: &ObjectData,
        mesh: &mut Mesh,
    ) {
        self.shader.use_program();
        self.position.enable();
        self.normal.enable();

        /*
         *
         * Setup camera and light.
         *
         */
        camera.upload(pass, &mut self.proj, &mut self.view);

        /*
         *
         * Setup object-related stuffs.
         *
         */
        let formated_transform = transform.to_homogeneous();
        let formated_scale =
            Matrix3::from_diagonal(&Vector3::new(scale.x, scale.y, scale.z));

        self.transform.upload(&formated_transform);
        self.scale.upload(&formated_scale);

        mesh.bind_coords(&mut self.position);
        mesh.bind_normals(&mut self.normal);
        mesh.bind_faces();

        Context::get().draw_elements(
            Context::TRIANGLES,
            mesh.num_pts() as i32,
            Context::UNSIGNED_SHORT,
            //VERTEX_INDEX_TYPE,
            0,
        );

        mesh.unbind();

        self.position.disable();
        self.normal.disable();
    }
}

static NORMAL_VERTEX_SRC: &str = include_str!("material.vert");
static NORMAL_FRAGMENT_SRC: &str = include_str!("material.frag");
