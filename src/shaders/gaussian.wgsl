struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    
    @location(0) @interpolate(flat) packed_size: u32
};

struct Splat {
    packed_pos: u32,
    packed_size: u32
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
}

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;

@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main(
    in: VertexInput
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
    );

    // unpack data from splats
    let actual_index = sort_indices[in.instance_index];
    let splat = splats[actual_index];
    let pos = unpack2x16float(splat.packed_pos);
    let size = unpack2x16float(splat.packed_size);

    let local = quad[in.vertex_index];
    let scaled_local = vec2(local.x * size.x, local.y * size.y);
    let world_pos = vec2(pos.x + scaled_local.x, pos.y + scaled_local.y);

    out.position = vec4<f32>(world_pos, 0.0, 1.0);
    out.packed_size = splat.packed_size;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = unpack2x16float(in.packed_size);
    let width = size.x;
    let height = size.y;

    return vec4<f32>(width, height, 0., 1.);
}