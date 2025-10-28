import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  render_settings_buffer: GPUBuffer,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const null_buffer = createBuffer(
    device,
    "null buffer",
    4,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data
  );

  const splat_buffer = createBuffer(
    device,
    "splat buffer",
    8 * pc.num_points,
    GPUBufferUsage.STORAGE,
    null
  );

  const render_settings_array = new Float32Array([1.0, pc.sh_deg]);
  const render_settings_buffer = createBuffer(
    device, 
    'render_settings_buffer', 
    8,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    render_settings_array
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  // bind groups for preprocess pipeline
  const camera_bind_group = device.createBindGroup({
    label: 'camera (gaussian)',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'gaussians (gaussian)',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort (gaussian)',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const splat_compute_bind_group = device.createBindGroup({
    label: 'splats (compute)',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer }},
      {binding: 1, resource: { buffer: render_settings_buffer }},
    ],
  })

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const drawArgs = new Uint32Array(4);
  drawArgs[0] = 6; // vertex count for instance
  drawArgs[1] = 0; // instance count. setting to 0 for now, will be updated by compute shader
  drawArgs[2] = 0; // First Vertex
  drawArgs[3] = 0; // First Instance

  const indirect_buffer = createBuffer(
    device,
    "indirect render buffer",
    16,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    drawArgs
  );

  // create render pipeline
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render pipeline',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    }
  });

  // bind group for splats
  const splat_render_bind_group = device.createBindGroup({
    label: 'splats (render)',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: { buffer: splat_buffer }},
      {binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  })

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  const doCompute = (encoder: GPUCommandEncoder) => {
    const computePass = encoder.beginComputePass();
    
    computePass.setPipeline(preprocess_pipeline);

    computePass.setBindGroup(0, camera_bind_group);
    computePass.setBindGroup(1, gaussian_bind_group);
    computePass.setBindGroup(2, sort_bind_group);
    computePass.setBindGroup(3, splat_compute_bind_group);

    computePass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));

    computePass.end();
  }

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      // reset sorting info buffer
      encoder.copyBufferToBuffer(
        null_buffer, 0,
        sorter.sort_info_buffer, 0,
        4
      );
      encoder.copyBufferToBuffer(
        null_buffer, 0,
        sorter.sort_dispatch_indirect_buffer, 0,
        4
      );

      doCompute(encoder);
      sorter.sort(encoder);

      // copy number of sorted points to indirect draw buffer
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 
        0, // keys_size offset 
        indirect_buffer, 
        4, // instance count offset
        4 
      );

      const pass = encoder.beginRenderPass({
        label: 'gaussian renderer',
        colorAttachments: [
          {
            view: texture_view,
            loadOp: 'clear',
            storeOp: 'store',
          }
        ],
      });

      pass.setPipeline(render_pipeline);

      pass.setBindGroup(0, splat_render_bind_group);

      pass.drawIndirect(indirect_buffer, 0);

      pass.end();
    },

    camera_buffer, render_settings_buffer
  };

}
