/// <reference types="@webgpu/types" />

export type GPUCtx = {
  device: GPUDevice;
  queue: GPUQueue;
};

export async function initGPU(): Promise<GPUCtx | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return { device, queue: device.queue };
  } catch (err) {
    console.warn('initGPU failed:', err);
    return null;
  }
}
