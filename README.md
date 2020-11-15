# Polyhedron Operators

This crate implements the [Conway Polyhedral
Operators](http://en.wikipedia.org/wiki/Conway_polyhedron_notation)
and their extensions by [George W. Hart](http://www.georgehart.com/)
and others.

This is an experiment to improve my understanding of iterators
in Rust. It is based on Hart’s OpenSCAD code which, being
functional, lends itself well to translation into functional Rust.

![Some brutalist Polyhedron, rendered with 3Delight|ɴsɪ](polyhedron.jpg)

## Supported Operators

- [x] **a** - ambo
- [x] **b** - bevel (equiv. to **ta**)
- [x] **c** - chamfer
- [x] **d** - dual
- [x] **e** - expand (a.k.a. explode, equiv. to **aa**)
- [x] **g** - gyro
- [x] **j** - join (equiv. to **dad**)
- [x] **M** - medial (equiv. to **dta**)
- [x] **m** - meta (equiv. to **k(3)j**)
- [x] **o** - ortho (equiv. to **jj**)
- [x] **p** - propellor
- [x] **k** - kis
- [x] **q** - quinto
- [x] **r** - reflect
- [x] **s** - snub (equiv. to **dgd**)
- [x] **t** - truncate vertices (equiv. to *dkNd*)
- [x] **w** - whirl
- [x] **z** - zip (equiv. to **dk**)

### Other Operators (TBD)

- [ ] **i** - inset/loft
- [ ] **h** - hexpropellor
- [ ] **l** - stellate
- [ ] **x** - extrude
- [ ] **z** - triangulate
- [ ] **H** - hollow (called ’intrude‘ in Wings3D)

## Playing

There is a playground example app to test things & have fun:

```
cargo run --release --example playground --features="nsi"
```

### Keyboard Commands

Use keys matching the operator name from the above list to apply.

Use `Up` and `Down` to adjust the parameter of the the last operator.
Combine with `Shift` for 10× the change.

`Delete` undoes the last (and only the last) operation.

Press `Enter` to render with 3Delight (requires a [3Delight|ɴsɪ
installation](https://www.3delight.com/download)).
Combine with `Shift` to render with 3Delight Cloud (requires
registration).

Press `Space` to save as `$HOME/polyhedron-<type>.obj`.

I use `kiss3d` for realtime preview which is close to the metal enough
to limit meshes to 65k vertices. This means the preview will be broken
if your mesh hits this limit.

Export & render will always yield a correct OBJ though. Which you can
view in Wings, Blender or another DCC app.

The app may crash though if your graphics driver doesn't handle such
ill-defined meshes gracefully. :)
