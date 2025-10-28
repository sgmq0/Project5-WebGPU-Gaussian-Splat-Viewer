const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    packed_pos: u32,
    packed_size: u32
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<storage, read_write> splats : array<Splat>;

@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.0);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // testing to acquire each resource from bind group 2
    let passes = sort_infos.passes;
    let depth = sort_depths[0];
    let index = sort_indices[0];
    let dispatch = sort_dispatch.dispatch_z;

    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.0);

    // transform into screen space
    let clip_pos = camera.proj * camera.view * pos;
    let ndc_pos = clip_pos.xy / clip_pos.w;

    // cull splats outside of view frustum
    if (ndc_pos.x < -1.2 || ndc_pos.x > 1.2 || ndc_pos.y < -1.2 || ndc_pos.y > 1.2 || clip_pos.w < 0.0) {
        return;
    }

    // compute 3d covariance
    let rot_a = unpack2x16float(vertex.rot[0]);
    let rot_b = unpack2x16float(vertex.rot[1]);
    let r = rot_a.x;
    let x = rot_a.y;
    let y = rot_b.x;
    let z = rot_b.y;
    
    let R = mat3x3f(
		1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
		2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
		2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
	);
    
    let scale_a = unpack2x16float(vertex.scale[0]);
    let scale_b = unpack2x16float(vertex.scale[1]);

    var S = mat3x3f();
    let scale_mod = render_settings.gaussian_scaling;
    // apparently have to use exponent here, not sure why tho
    S[0][0] = exp(scale_a.x) * scale_mod;
    S[1][1] = exp(scale_a.y) * scale_mod;
    S[2][2] = exp(scale_b.x) * scale_mod;
    
    let M = S * R;
	let Sigma = transpose(M) * M; // 3d covariance
    
    // compute 2d covariance
    var t = (camera.view * pos).xyz;
    let focal_x = camera.focal.x;
    let focal_y = camera.focal.y;

    let limx = 0.65 * camera.viewport.x / focal_x;
    let limy = 0.65 * camera.viewport.y / focal_y;
	let txtz = t.x / t.z;
	let tytz = t.y / t.z; 

    t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat3x3f(
		focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
		0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0, 0.0, 0.0
    );

    let W = mat3x3f(
		camera.view[0][0], camera.view[0][1], camera.view[0][2],
		camera.view[1][0], camera.view[1][1], camera.view[1][2],
		camera.view[2][0], camera.view[2][1], camera.view[2][2]
    );

    let T = W * J;

    let Vrk = mat3x3f(
		Sigma[0][0], Sigma[0][1], Sigma[0][2],
		Sigma[1][0], Sigma[1][1], Sigma[1][2],
		Sigma[2][0], Sigma[2][1], Sigma[2][2]
    );

    var cov_mat = transpose(T) * transpose(Vrk) * T;
	cov_mat[0][0] += 0.3;
	cov_mat[1][1] += 0.3;

    let cov = vec3f(cov_mat[0][0], cov_mat[0][1], cov_mat[1][1]); // 2d covariance
    
    // compute radius
    let det = (cov.x * cov.z - cov.y * cov.y);
    if (det <= 0) {
        return;
    }
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    let size = vec2f(radius, radius) / camera.viewport;

    // increment the index for splat storage
    let atomic_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // store data into splats
    splats[atomic_idx].packed_pos = pack2x16float(ndc_pos);
    splats[atomic_idx].packed_size = pack2x16float(size);

    // store data into sort stuff
    sort_indices[atomic_idx] = atomic_idx;
    let view_depth = (camera.view * pos).z;
    sort_depths[atomic_idx] = bitcast<u32>(100.0f - view_depth);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (atomic_idx % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}