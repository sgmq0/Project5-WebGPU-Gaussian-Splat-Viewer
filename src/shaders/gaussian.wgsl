struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    
    @location(0) @interpolate(flat) packed_pos: u32,
    @location(1) @interpolate(flat) packed_size: u32,
    @location(2) color: vec3<f32>,
    @location(3) conic_opacity: vec4<f32>,
    @location(4) conic_center: vec2<f32>,
};

struct Splat {
    packed_pos_size: array<u32,2>,
    packed_color: array<u32,2>,
    packed_conic_opacity: array<u32,2>
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

@group(1) @binding(0)
var<uniform> camera: CameraUniforms;

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
    let pos = unpack2x16float(splat.packed_pos_size[0]);
    let size = unpack2x16float(splat.packed_pos_size[1]);
    let color_rg = unpack2x16float(splat.packed_color[0]);
    let color_ba = unpack2x16float(splat.packed_color[1]);

    let local = quad[in.vertex_index];
    let scaled_local = vec2(local.x * size.x, local.y * size.y);
    let world_pos = vec2(pos.x + scaled_local.x, pos.y + scaled_local.y);

    // do all the conic stuff
    let conic_xy = unpack2x16float(splat.packed_conic_opacity[0]);
    let conic_za = unpack2x16float(splat.packed_conic_opacity[1]);
    let conic_opacity = vec4f(conic_xy.x, conic_xy.y, conic_za.x, conic_za.y);
    let conic_center = vec2f(pos.x, pos.y);

    out.position = vec4<f32>(world_pos, 0.0, 1.0);
    out.packed_pos = splat.packed_pos_size[0];
    out.packed_size = splat.packed_pos_size[1];
    out.conic_opacity = conic_opacity;
    out.conic_center = conic_center;
    out.color = vec3<f32>(color_rg.x, color_rg.y, color_ba.x);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pos = unpack2x16float(in.packed_pos);
    let size = unpack2x16float(in.packed_size);

    // compute ndc pos of vertex
    var ndc = in.position.xy / camera.viewport * 2.0 - 1.0;
    ndc.y *= -1.0;

    // compute offset
    var offset = ndc - in.conic_center.xy;
    offset.x *= -1.0;
    offset = offset * camera.viewport * 0.5;

    // compute the power
    let o = in.conic_opacity;
    let power = -0.5 * (o.x * offset.x * offset.x + o.z * offset.y * offset.y) - o.y * offset.x * offset.y;

    if (power > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // compute the alpha
    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

     return vec4<f32>(in.color * alpha, alpha);
}