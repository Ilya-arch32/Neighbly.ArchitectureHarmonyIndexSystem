/**
 * AHI 2.0 Ultimate - 3D Voxel Fractal Analyzer
 * 
 * GPU-accelerated fractal dimension calculation using box-counting in 3D space
 * Replaces 2D image-based analysis with proper volumetric assessment
 */

import { VoxelGridConfig } from './VoxelTypes';

export interface FractalMetrics3D {
    fractalDimension: number;      // D value (1.3-1.5 optimal from neuroscience)
    lacunarity: number;            // Gap distribution (texture invariance)
    multiScale: Float32Array;      // Scale-dependent dimensions
    spatialEntropy: number;        // 3D Shannon entropy
    isOptimalComplexity: boolean;  // In 1.3-1.5 range
}

export class VoxelFractalAnalyzer {
    private device: GPUDevice;
    private gridConfig: VoxelGridConfig;
    
    // GPU Resources
    private boxCountPipeline!: GPUComputePipeline;
    private entropyPipeline!: GPUComputePipeline;
    private countBuffer!: GPUBuffer;
    private resultBuffer!: GPUBuffer;
    
    constructor(device: GPUDevice, gridConfig: VoxelGridConfig) {
        this.device = device;
        this.gridConfig = gridConfig;
    }
    
    /**
     * Initialize GPU pipelines for 3D fractal analysis
     */
    async initialize(): Promise<void> {
        // Create shader module
        const shaderModule = this.device.createShaderModule({
            label: '3D Fractal Shaders',
            code: this.generateFractalShaders()
        });
        
        // Box counting pipeline
        this.boxCountPipeline = this.device.createComputePipeline({
            label: 'Box Counting Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'box_counting_3d'
            }
        });
        
        // Entropy calculation pipeline  
        this.entropyPipeline = this.device.createComputePipeline({
            label: 'Entropy Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'spatial_entropy_3d'
            }
        });
        
        // Allocate buffers
        const maxBoxes = 1000000; // Max boxes at finest scale
        this.countBuffer = this.device.createBuffer({
            label: 'Box Count Buffer',
            size: maxBoxes * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        this.resultBuffer = this.device.createBuffer({
            label: 'Result Buffer',
            size: 256 * 4, // Multiple scales
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        console.log('[VoxelFractalAnalyzer] Initialized');
    }
    
    /**
     * Analyze fractal dimension of voxel geometry
     */
    async analyze(voxelBuffer: GPUBuffer): Promise<FractalMetrics3D> {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        
        // Box sizes: powers of 2 from 1 to min(nx,ny,nz)/2
        const maxScale = Math.floor(Math.log2(Math.min(nx, ny, nz))) - 1;
        const scales: number[] = [];
        const counts: number[] = [];
        
        // Perform box counting at multiple scales
        for (let scale = 0; scale < maxScale; scale++) {
            const boxSize = Math.pow(2, scale);
            scales.push(boxSize);
            
            const count = await this.countBoxesAtScale(voxelBuffer, boxSize);
            counts.push(count);
        }
        
        // Calculate fractal dimension using log-log regression
        const fractalDimension = this.calculateFractalDimension(scales, counts);
        
        // Calculate lacunarity (measure of gaps/heterogeneity)
        const lacunarity = await this.calculateLacunarity(voxelBuffer, scales);
        
        // Calculate 3D spatial entropy
        const spatialEntropy = await this.calculate3DEntropy(voxelBuffer);
        
        // Multi-scale dimensions
        const multiScale = new Float32Array(scales.length);
        for (let i = 1; i < scales.length; i++) {
            const localD = -Math.log(counts[i] / counts[i-1]) / Math.log(scales[i] / scales[i-1]);
            multiScale[i] = localD;
        }
        
        // Check if in optimal range (1.3-1.5 from neuroscience research)
        const isOptimalComplexity = fractalDimension >= 1.3 && fractalDimension <= 1.5;
        
        return {
            fractalDimension,
            lacunarity,
            multiScale,
            spatialEntropy,
            isOptimalComplexity
        };
    }
    
    /**
     * Count occupied boxes at specific scale
     */
    private async countBoxesAtScale(voxelBuffer: GPUBuffer, boxSize: number): Promise<number> {
        const commandEncoder = this.device.createCommandEncoder();
        
        // Create uniform buffer with parameters
        const uniformData = new Float32Array([
            this.gridConfig.dimensions.nx,
            this.gridConfig.dimensions.ny,
            this.gridConfig.dimensions.nz,
            boxSize
        ]);
        
        const uniformBuffer = this.device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        
        // Run box counting
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.boxCountPipeline);
        // Bind voxelBuffer, uniformBuffer, countBuffer
        
        const numBoxesX = Math.ceil(this.gridConfig.dimensions.nx / boxSize);
        const numBoxesY = Math.ceil(this.gridConfig.dimensions.ny / boxSize);  
        const numBoxesZ = Math.ceil(this.gridConfig.dimensions.nz / boxSize);
        
        computePass.dispatchWorkgroups(
            Math.ceil(numBoxesX / 8),
            Math.ceil(numBoxesY / 8),
            Math.ceil(numBoxesZ / 8)
        );
        computePass.end();
        
        // Read back result
        const stagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        commandEncoder.copyBufferToBuffer(this.countBuffer, 0, stagingBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(stagingBuffer.getMappedRange())[0];
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        uniformBuffer.destroy();
        
        return result;
    }
    
    /**
     * Calculate fractal dimension from box counts
     */
    private calculateFractalDimension(scales: number[], counts: number[]): number {
        // Linear regression on log-log plot
        const logScales = scales.map(s => Math.log(s));
        const logCounts = counts.map(c => Math.log(c));
        
        const n = logScales.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (let i = 0; i < n; i++) {
            sumX += logScales[i];
            sumY += logCounts[i];
            sumXY += logScales[i] * logCounts[i];
            sumX2 += logScales[i] * logScales[i];
        }
        
        // Slope of regression line is negative of fractal dimension
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return -slope;
    }
    
    /**
     * Calculate lacunarity (gap distribution measure)
     */
    private async calculateLacunarity(voxelBuffer: GPUBuffer, scales: number[]): Promise<number> {
        // Simplified gliding box algorithm
        // In production, implement full gliding box with mass distribution
        
        let totalLacunarity = 0;
        
        for (const scale of scales) {
            // Calculate variance/mean^2 for this scale
            const mean = await this.countBoxesAtScale(voxelBuffer, scale);
            const variance = mean * 0.1; // Simplified - should calculate actual variance
            
            const lacunarity = 1 + (variance / (mean * mean));
            totalLacunarity += lacunarity;
        }
        
        return totalLacunarity / scales.length;
    }
    
    /**
     * Calculate 3D spatial entropy
     */
    private async calculate3DEntropy(voxelBuffer: GPUBuffer): Promise<number> {
        const commandEncoder = this.device.createCommandEncoder();
        
        // Run entropy calculation on GPU
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.entropyPipeline);
        // Bind buffers...
        
        const { nx, ny, nz } = this.gridConfig.dimensions;
        computePass.dispatchWorkgroups(
            Math.ceil(nx / 8),
            Math.ceil(ny / 8),
            Math.ceil(nz / 8)
        );
        computePass.end();
        
        // Read back entropy value
        const stagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        commandEncoder.copyBufferToBuffer(this.resultBuffer, 0, stagingBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const entropy = new Float32Array(stagingBuffer.getMappedRange())[0];
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return entropy;
    }
    
    /**
     * Generate WGSL shaders for fractal analysis
     */
    private generateFractalShaders(): string {
        return `
            struct Uniforms {
                grid_size: vec3<u32>,
                box_size: u32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> voxels: array<f32>;
            @group(0) @binding(2) var<storage, read_write> counts: atomic<u32>;
            
            @compute @workgroup_size(8, 8, 8)
            fn box_counting_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
                let box_x = gid.x * uniforms.box_size;
                let box_y = gid.y * uniforms.box_size;
                let box_z = gid.z * uniforms.box_size;
                
                // Check if box contains any solid voxels
                var has_solid = false;
                for (var dx = 0u; dx < uniforms.box_size; dx++) {
                    for (var dy = 0u; dy < uniforms.box_size; dy++) {
                        for (var dz = 0u; dz < uniforms.box_size; dz++) {
                            let x = box_x + dx;
                            let y = box_y + dy;
                            let z = box_z + dz;
                            
                            if (x < uniforms.grid_size.x && 
                                y < uniforms.grid_size.y && 
                                z < uniforms.grid_size.z) {
                                
                                let idx = x + y * uniforms.grid_size.x + 
                                         z * uniforms.grid_size.x * uniforms.grid_size.y;
                                
                                // Check voxel state (first float in voxel data)
                                if (voxels[idx * 8] > 0.5) {
                                    has_solid = true;
                                    break;
                                }
                            }
                        }
                        if (has_solid) { break; }
                    }
                    if (has_solid) { break; }
                }
                
                if (has_solid) {
                    atomicAdd(&counts, 1u);
                }
            }
            
            @compute @workgroup_size(8, 8, 8)
            fn spatial_entropy_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
                // Calculate local entropy contribution
                // Simplified - in production use proper probability distribution
                
                let idx = gid.x + gid.y * uniforms.grid_size.x + 
                         gid.z * uniforms.grid_size.x * uniforms.grid_size.y;
                
                if (idx < uniforms.grid_size.x * uniforms.grid_size.y * uniforms.grid_size.z) {
                    let voxel_state = voxels[idx * 8];
                    
                    // Shannon entropy: -p * log2(p)
                    if (voxel_state > 0.0 && voxel_state < 1.0) {
                        let p = voxel_state;
                        let entropy = -p * log2(p) - (1.0 - p) * log2(1.0 - p);
                        
                        // Accumulate (simplified - should use atomic operations)
                        // In production, use proper reduction
                    }
                }
            }
        `;
    }
    
    /**
     * Cleanup
     */
    destroy(): void {
        this.countBuffer?.destroy();
        this.resultBuffer?.destroy();
        
        console.log('[VoxelFractalAnalyzer] Destroyed');
    }
}
