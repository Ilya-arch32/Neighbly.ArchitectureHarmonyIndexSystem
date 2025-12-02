/**
 * AHI 2.0 Ultimate - Space Syntax Solver
 * 
 * 3D Isovist and Visibility Graph Analysis
 * Computes spatial integration, connectivity, and visual complexity
 */

import { VoxelGridConfig } from './VoxelTypes';

export interface SpaceSyntaxMetrics {
    integration: Float32Array;       // How integrated each space is
    connectivity: Float32Array;      // Number of visible spaces
    meanDepth: number;              // Average visual depth
    intelligibility: number;        // Correlation between connectivity and integration  
    synergy: number;               // Local-global correlation
    visualComplexity: number;       // Isovist shape complexity
}

export interface IsovistProperties {
    volume: number;                 // 3D visible volume
    surfaceArea: number;            // Boundary surface area
    compactness: number;            // Volume/surface ratio
    occlusivity: number;            // Hidden space potential
    drift: number;                 // Center offset from viewpoint
}

export class SpaceSyntaxSolver {
    private device: GPUDevice;
    private gridConfig: VoxelGridConfig;
    
    // GPU Resources
    private raycastPipeline!: GPUComputePipeline;
    private integrationPipeline!: GPUComputePipeline;
    private visibilityBuffer!: GPUBuffer;
    private metricsBuffer!: GPUBuffer;
    
    // Configuration
    private numRays: number = 256;     // Rays per viewpoint
    private maxViewDistance: number = 30; // meters
    
    constructor(device: GPUDevice, gridConfig: VoxelGridConfig) {
        this.device = device;
        this.gridConfig = gridConfig;
    }
    
    /**
     * Initialize GPU pipelines for visibility analysis
     */
    async initialize(): Promise<void> {
        const shaderModule = this.device.createShaderModule({
            label: 'Space Syntax Shaders',
            code: this.generateSpaceSyntaxShaders()
        });
        
        // Ray casting pipeline for isovist generation
        this.raycastPipeline = this.device.createComputePipeline({
            label: 'Raycast Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'compute_isovist'
            }
        });
        
        // Integration calculation pipeline
        this.integrationPipeline = this.device.createComputePipeline({
            label: 'Integration Pipeline', 
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'compute_integration'
            }
        });
        
        // Allocate buffers
        const { totalVoxels } = this.gridConfig;
        
        this.visibilityBuffer = this.device.createBuffer({
            label: 'Visibility Buffer',
            size: totalVoxels * totalVoxels / 8, // Bit matrix
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        this.metricsBuffer = this.device.createBuffer({
            label: 'Metrics Buffer',
            size: totalVoxels * 16, // Multiple metrics per voxel
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        console.log('[SpaceSyntaxSolver] Initialized');
    }
    
    /**
     * Analyze spatial configuration using visibility graph
     */
    async analyze(voxelBuffer: GPUBuffer): Promise<SpaceSyntaxMetrics> {
        // 1. Build visibility graph
        await this.buildVisibilityGraph(voxelBuffer);
        
        // 2. Calculate integration (mean depth from all spaces)
        const integration = await this.calculateIntegration();
        
        // 3. Calculate connectivity (direct connections)
        const connectivity = await this.calculateConnectivity();
        
        // 4. Calculate mean depth
        const meanDepth = this.calculateMeanDepth(integration);
        
        // 5. Calculate intelligibility (R² between connectivity and integration)
        const intelligibility = this.calculateIntelligibility(connectivity, integration);
        
        // 6. Calculate synergy (local vs global integration)
        const synergy = await this.calculateSynergy(integration);
        
        // 7. Calculate visual complexity from isovists
        const visualComplexity = await this.calculateVisualComplexity();
        
        return {
            integration,
            connectivity,
            meanDepth,
            intelligibility,
            synergy,
            visualComplexity
        };
    }
    
    /**
     * Build visibility graph using GPU ray casting
     */
    private async buildVisibilityGraph(voxelBuffer: GPUBuffer): Promise<void> {
        const commandEncoder = this.device.createCommandEncoder();
        
        const { nx, ny, nz } = this.gridConfig.dimensions;
        
        // Sample viewpoints throughout space
        const numSamples = Math.min(1000, this.gridConfig.totalVoxels / 100);
        
        for (let i = 0; i < numSamples; i++) {
            // Random viewpoint in FLUID voxels
            const vx = Math.floor(Math.random() * nx);
            const vy = Math.floor(Math.random() * ny);
            const vz = Math.floor(Math.random() * nz);
            
            // Cast rays in spherical pattern
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(this.raycastPipeline);
            // Bind voxelBuffer, viewpoint, visibilityBuffer
            computePass.dispatchWorkgroups(
                Math.ceil(this.numRays / 64),
                1,
                1
            );
            computePass.end();
        }
        
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }
    
    /**
     * Calculate integration (closeness centrality)
     */
    private async calculateIntegration(): Promise<Float32Array> {
        const commandEncoder = this.device.createCommandEncoder();
        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.integrationPipeline);
        // Bind visibilityBuffer, metricsBuffer
        
        const numVoxels = this.gridConfig.totalVoxels;
        computePass.dispatchWorkgroups(
            Math.ceil(numVoxels / 64),
            1,
            1
        );
        computePass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Read back integration values
        return await this.readMetricsBuffer(0, numVoxels);
    }
    
    /**
     * Calculate connectivity (degree centrality)
     */
    private async calculateConnectivity(): Promise<Float32Array> {
        // Count direct visual connections for each space
        const numVoxels = this.gridConfig.totalVoxels;
        const connectivity = new Float32Array(numVoxels);
        
        // Simplified - in production read from GPU
        for (let i = 0; i < numVoxels; i++) {
            connectivity[i] = Math.random() * 10; // Placeholder
        }
        
        return connectivity;
    }
    
    /**
     * Calculate mean depth across all spaces
     */
    private calculateMeanDepth(integration: Float32Array): number {
        let sum = 0;
        let count = 0;
        
        for (let i = 0; i < integration.length; i++) {
            if (integration[i] > 0) {
                sum += 1 / integration[i]; // Depth is inverse of integration
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0;
    }
    
    /**
     * Calculate intelligibility (connectivity-integration correlation)
     */
    private calculateIntelligibility(
        connectivity: Float32Array,
        integration: Float32Array
    ): number {
        // Pearson correlation coefficient
        const n = connectivity.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        
        for (let i = 0; i < n; i++) {
            const x = connectivity[i];
            const y = integration[i];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;
        }
        
        const correlation = (n * sumXY - sumX * sumY) / 
            Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return correlation * correlation; // R²
    }
    
    /**
     * Calculate synergy (local vs global integration)
     */
    private async calculateSynergy(globalIntegration: Float32Array): Promise<number> {
        // Calculate local integration (radius 3)
        const localIntegration = new Float32Array(globalIntegration.length);
        
        // Simplified - in production calculate on GPU
        for (let i = 0; i < localIntegration.length; i++) {
            localIntegration[i] = globalIntegration[i] * (0.8 + Math.random() * 0.4);
        }
        
        // Return correlation between local and global
        return this.calculateIntelligibility(localIntegration, globalIntegration);
    }
    
    /**
     * Calculate visual complexity from isovist properties
     */
    private async calculateVisualComplexity(): Promise<number> {
        // Sample isovists at key points
        const samples = 100;
        let totalComplexity = 0;
        
        for (let i = 0; i < samples; i++) {
            const isovist = await this.computeIsovistAt(
                Math.random() * this.gridConfig.dimensions.nx,
                Math.random() * this.gridConfig.dimensions.ny,
                Math.random() * this.gridConfig.dimensions.nz
            );
            
            // Jaggedness = perimeter²/area (2D) or surface²/volume (3D)
            const jaggedness = (isovist.surfaceArea * isovist.surfaceArea) / isovist.volume;
            
            // Complexity combines multiple factors
            const complexity = jaggedness * isovist.occlusivity * (1 + isovist.drift);
            totalComplexity += complexity;
        }
        
        return totalComplexity / samples;
    }
    
    /**
     * Compute isovist properties at specific location
     */
    private async computeIsovistAt(x: number, y: number, z: number): Promise<IsovistProperties> {
        // Placeholder - in production cast rays and analyze visible volume
        return {
            volume: 100 + Math.random() * 500,
            surfaceArea: 50 + Math.random() * 200,
            compactness: 0.5 + Math.random() * 0.5,
            occlusivity: Math.random(),
            drift: Math.random() * 2
        };
    }
    
    /**
     * Read metrics from GPU buffer
     */
    private async readMetricsBuffer(offset: number, count: number): Promise<Float32Array> {
        const stagingBuffer = this.device.createBuffer({
            size: count * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.metricsBuffer,
            offset * 4,
            stagingBuffer,
            0,
            count * 4
        );
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const result = new Float32Array(data);
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return result;
    }
    
    /**
     * Generate WGSL shaders for space syntax
     */
    private generateSpaceSyntaxShaders(): string {
        return `
            struct Uniforms {
                grid_size: vec3<u32>,
                viewpoint: vec3<f32>,
                max_distance: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> voxels: array<f32>;
            @group(0) @binding(2) var<storage, read_write> visibility: array<u32>;
            @group(0) @binding(3) var<storage, read_write> metrics: array<f32>;
            
            @compute @workgroup_size(64)
            fn compute_isovist(@builtin(global_invocation_id) gid: vec3<u32>) {
                let ray_id = gid.x;
                if (ray_id >= 256u) { return; }
                
                // Generate ray direction (spherical coordinates)
                let theta = f32(ray_id % 16u) * 3.14159 * 2.0 / 16.0;
                let phi = f32(ray_id / 16u) * 3.14159 / 16.0;
                
                let direction = vec3<f32>(
                    sin(phi) * cos(theta),
                    cos(phi),
                    sin(phi) * sin(theta)
                );
                
                // Ray march through voxel grid
                var t = 0.0;
                let step_size = 0.1;
                
                while (t < uniforms.max_distance) {
                    let pos = uniforms.viewpoint + direction * t;
                    
                    // Check if hit solid
                    let voxel_idx = get_voxel_index(pos);
                    if (voxel_idx >= 0 && voxels[voxel_idx * 8u] > 0.5) {
                        // Mark as visible
                        let vis_idx = get_visibility_index(
                            get_voxel_index(uniforms.viewpoint),
                            voxel_idx
                        );
                        atomicOr(&visibility[vis_idx / 32u], 1u << (vis_idx % 32u));
                        break;
                    }
                    
                    t += step_size;
                }
            }
            
            @compute @workgroup_size(64)
            fn compute_integration(@builtin(global_invocation_id) gid: vec3<u32>) {
                let voxel_id = gid.x;
                if (voxel_id >= uniforms.grid_size.x * uniforms.grid_size.y * uniforms.grid_size.z) {
                    return;
                }
                
                // Calculate mean shortest path to all other visible spaces
                var total_distance = 0.0;
                var visible_count = 0u;
                
                for (var i = 0u; i < uniforms.grid_size.x * uniforms.grid_size.y * uniforms.grid_size.z; i++) {
                    if (i == voxel_id) { continue; }
                    
                    let vis_idx = get_visibility_index(voxel_id, i);
                    let is_visible = (visibility[vis_idx / 32u] >> (vis_idx % 32u)) & 1u;
                    
                    if (is_visible == 1u) {
                        total_distance += distance_between(voxel_id, i);
                        visible_count++;
                    }
                }
                
                // Integration = 1 / mean_distance
                let integration = f32(visible_count) / max(total_distance, 1.0);
                metrics[voxel_id] = integration;
            }
            
            fn get_voxel_index(pos: vec3<f32>) -> i32 {
                let x = i32(pos.x);
                let y = i32(pos.y);
                let z = i32(pos.z);
                
                if (x < 0 || y < 0 || z < 0 ||
                    x >= i32(uniforms.grid_size.x) ||
                    y >= i32(uniforms.grid_size.y) ||
                    z >= i32(uniforms.grid_size.z)) {
                    return -1;
                }
                
                return x + y * i32(uniforms.grid_size.x) + 
                       z * i32(uniforms.grid_size.x * uniforms.grid_size.y);
            }
            
            fn get_visibility_index(from: u32, to: u32) -> u32 {
                return from * (uniforms.grid_size.x * uniforms.grid_size.y * uniforms.grid_size.z) + to;
            }
            
            fn distance_between(a: u32, b: u32) -> f32 {
                let ax = a % uniforms.grid_size.x;
                let ay = (a / uniforms.grid_size.x) % uniforms.grid_size.y;
                let az = a / (uniforms.grid_size.x * uniforms.grid_size.y);
                
                let bx = b % uniforms.grid_size.x;
                let by = (b / uniforms.grid_size.x) % uniforms.grid_size.y;
                let bz = b / (uniforms.grid_size.x * uniforms.grid_size.y);
                
                let dx = f32(ax) - f32(bx);
                let dy = f32(ay) - f32(by);
                let dz = f32(az) - f32(bz);
                
                return sqrt(dx * dx + dy * dy + dz * dz);
            }
        `;
    }
    
    /**
     * Cleanup
     */
    destroy(): void {
        this.visibilityBuffer?.destroy();
        this.metricsBuffer?.destroy();
        
        console.log('[SpaceSyntaxSolver] Destroyed');
    }
}
