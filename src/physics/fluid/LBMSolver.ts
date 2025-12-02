/**
 * AHI 2.0 Ultimate - LBM Solver TypeScript Wrapper
 * 
 * Управляет WebGPU compute pipeline для LBM D3Q19 симуляции
 */

import { VoxelGridConfig, SimulationSnapshot, PHYSICS_CONSTANTS } from './VoxelTypes';
import lbmShaderCode from './lbm_solver.wgsl?raw';

export interface LBMConfig {
    tau: number;                // Relaxation time (связан с вязкостью)
    nu: number;                 // Kinematic viscosity (м²/с)
    rho0: number;               // Reference density (кг/м³)
    gravity: [number, number, number]; // Gravity vector (м/с²)
    dt: number;                 // Time step (с) - initial value, will be adapted
    enableBuoyancy: boolean;    // Тепловая плавучесть
    beta: number;               // Thermal expansion coefficient (1/К)
    smagorinskyConstant: number; // Константа Смагоринского C_s (0.1-0.2) для LES
    enableLES: boolean;         // Включение LES модели турбулентности
    
    // TRL 7: Adaptive time stepping parameters
    enableAdaptiveDt: boolean;   // Включить адаптивный шаг времени
    maxMach: number;             // Максимальное число Маха (0.1 для несжимаемого LBM)
    cflFactor: number;           // CFL safety factor (0.5-0.9)
    dtMin: number;               // Минимальный dt (с)
    dtMax: number;               // Максимальный dt (с)
    dtUpdateInterval: number;    // Интервал обновления dt (шагов)
}

/**
 * TRL 7: Stability metrics for adaptive time stepping
 */
export interface StabilityMetrics {
    currentDt: number;           // Текущий шаг времени (с)
    maxVelocity: number;         // Максимальная скорость в поле (м/с)
    machNumber: number;          // Текущее число Маха
    cflNumber: number;           // Текущее число CFL
    isStable: boolean;           // Симуляция стабильна
    stabilityMargin: number;     // Запас стабильности (0-1)
}

export class LBMSolver {
    private device: GPUDevice;
    private gridConfig: VoxelGridConfig;
    private config: LBMConfig;
    
    // GPU Buffers
    private uniformBuffer!: GPUBuffer;
    private fInBuffer!: GPUBuffer;
    private fOutBuffer!: GPUBuffer;
    private densityBuffer!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    private temperatureBuffer!: GPUBuffer;
    private voxelStateBuffer!: GPUBuffer;
    
    // Compute pipelines
    private collisionPipeline!: GPUComputePipeline;
    private streamingPipeline!: GPUComputePipeline;
    private inletBCPipeline!: GPUComputePipeline;
    private outletBCPipeline!: GPUComputePipeline;
    
    private bindGroup!: GPUBindGroup;
    
    private currentStep: number = 0;
    private initialized: boolean = false;
    
    // TRL 7: Adaptive time stepping state
    private currentDt: number = 0.001;
    private lastMaxVelocity: number = 0;
    private lastMachNumber: number = 0;
    private dtHistory: number[] = [];
    
    // LBM lattice constants
    private readonly CS = 1.0 / Math.sqrt(3);  // Lattice speed of sound
    
    constructor(device: GPUDevice, gridConfig: VoxelGridConfig, config?: Partial<LBMConfig>) {
        this.device = device;
        this.gridConfig = gridConfig;
        
        // Default configuration
        this.config = {
            tau: 0.6,
            nu: 1.5e-5, // Air at 20°C
            rho0: PHYSICS_CONSTANTS.AIR_DENSITY,
            gravity: [0, 0, -9.81],
            dt: 0.001, // 1ms
            enableBuoyancy: true,
            beta: 3.4e-3, // Air thermal expansion
            smagorinskyConstant: 0.15, // Стандартное значение для зданий (0.1-0.2)
            enableLES: true, // Включаем LES по умолчанию для высоких Re
            // TRL 7: Adaptive time stepping defaults
            enableAdaptiveDt: true,
            maxMach: 0.1,      // Critical for incompressible LBM validity
            cflFactor: 0.7,    // Conservative CFL factor
            dtMin: 1e-6,       // 1 microsecond minimum
            dtMax: 0.01,       // 10 millisecond maximum
            dtUpdateInterval: 10,  // Update dt every 10 steps
            ...config,
        };
        
        this.currentDt = this.config.dt;
        
        console.log('[LBMSolver] Initialized with config:', this.config);
    }
    
    /**
     * Инициализация WebGPU ресурсов
     */
    async initialize(voxelStateData: Float32Array, initialTemperature: Float32Array): Promise<void> {
        console.time('[LBMSolver] Initialization');
        
        const { nx, ny, nz, totalVoxels } = this.gridConfig.dimensions;
        const D3Q19_DIRECTIONS = 19;
        
        // Создаем буферы
        this.createBuffers(totalVoxels, D3Q19_DIRECTIONS, voxelStateData, initialTemperature);
        
        // Компилируем шейдеры
        const shaderModule = this.device.createShaderModule({
            label: 'LBM D3Q19 Shader',
            code: lbmShaderCode,
        });
        
        // Создаем bind group layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'LBM Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });
        
        // Создаем pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'LBM Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout],
        });
        
        // Создаем compute pipelines
        this.collisionPipeline = this.device.createComputePipeline({
            label: 'LBM Collision Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'collisionStep',
            },
        });
        
        this.streamingPipeline = this.device.createComputePipeline({
            label: 'LBM Streaming Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'streamingStep',
            },
        });
        
        this.inletBCPipeline = this.device.createComputePipeline({
            label: 'LBM Inlet BC Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'applyInletBC',
            },
        });
        
        this.outletBCPipeline = this.device.createComputePipeline({
            label: 'LBM Outlet BC Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'applyOutletBC',
            },
        });
        
        // Создаем bind group
        this.bindGroup = this.device.createBindGroup({
            label: 'LBM Bind Group',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.fInBuffer } },
                { binding: 2, resource: { buffer: this.fOutBuffer } },
                { binding: 3, resource: { buffer: this.densityBuffer } },
                { binding: 4, resource: { buffer: this.velocityBuffer } },
                { binding: 5, resource: { buffer: this.temperatureBuffer } },
                { binding: 6, resource: { buffer: this.voxelStateBuffer } },
            ],
        });
        
        this.initialized = true;
        console.timeEnd('[LBMSolver] Initialization');
        console.log(`[LBMSolver] Ready for simulation (${totalVoxels} voxels, ${D3Q19_DIRECTIONS} velocities)`);
    }
    
    /**
     * Создание GPU буферов
     */
    private createBuffers(
        totalVoxels: number,
        directions: number,
        voxelStateData: Float32Array,
        initialTemperature: Float32Array
    ): void {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        
        // Uniform buffer (параметры симуляции)
        // Структура соответствует SimulationParams в WGSL:
        // - nx, ny, nz, resolution (4 x f32)
        // - tau, omega (2 x f32)
        // - rho0, nu (2 x f32)
        // - gravity (vec3<f32>) + dt (1 x f32 padding to vec4)
        // - enableBuoyancy, beta (2 x f32)
        // - smagorinskyConstant, enableLES (2 x f32)
        const uniformData = new Float32Array([
            nx, ny, nz, this.gridConfig.resolution,
            this.config.tau,
            1.0 / this.config.tau, // omega
            this.config.rho0,
            this.config.nu,
            ...this.config.gravity, // gravity vec3
            this.config.dt,         // dt (заполняет vec4)
            this.config.enableBuoyancy ? 1 : 0,
            this.config.beta,
            this.config.smagorinskyConstant, // Константа Смагоринского
            this.config.enableLES ? 1 : 0,   // Флаг LES
        ]);
        
        this.uniformBuffer = this.device.createBuffer({
            label: 'LBM Uniform Buffer',
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
        
        // Distribution functions (f_in, f_out)
        const fSize = totalVoxels * directions * 4; // float32
        
        this.fInBuffer = this.device.createBuffer({
            label: 'LBM f_in Buffer',
            size: fSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        
        this.fOutBuffer = this.device.createBuffer({
            label: 'LBM f_out Buffer',
            size: fSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        
        // Инициализация equilibrium distribution
        const initialF = new Float32Array(totalVoxels * directions);
        for (let i = 0; i < totalVoxels; i++) {
            // f_eq для покоя: rho * w_i
            initialF[i * directions + 0] = this.config.rho0 * (1.0/3.0); // rest particle
            for (let q = 1; q < directions; q++) {
                initialF[i * directions + q] = this.config.rho0 * (q < 7 ? 1.0/18.0 : 1.0/36.0);
            }
        }
        this.device.queue.writeBuffer(this.fInBuffer, 0, initialF);
        this.device.queue.writeBuffer(this.fOutBuffer, 0, initialF);
        
        // Macroscopic fields
        this.densityBuffer = this.device.createBuffer({
            label: 'LBM Density Buffer',
            size: totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        
        this.velocityBuffer = this.device.createBuffer({
            label: 'LBM Velocity Buffer',
            size: totalVoxels * 3 * 4, // vec3<f32>
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        
        this.temperatureBuffer = this.device.createBuffer({
            label: 'LBM Temperature Buffer',
            size: initialTemperature.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.device.queue.writeBuffer(this.temperatureBuffer, 0, initialTemperature);
        
        this.voxelStateBuffer = this.device.createBuffer({
            label: 'Voxel State Buffer',
            size: voxelStateData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.voxelStateBuffer, 0, voxelStateData);
    }
    
    /**
     * TRL 7: Calculate adaptive time step based on CFL and Mach number constraints
     * 
     * CFL condition: dt ≤ CFL_factor × dx / u_max
     * Mach constraint: u_max / c_s ≤ Ma_max  =>  dt ≤ Ma_max × c_s × dx / u_max
     * 
     * For LBM: c_s = 1/sqrt(3) in lattice units, physical c_s = dx/dt × 1/sqrt(3)
     */
    async calculateAdaptiveDt(): Promise<number> {
        if (!this.config.enableAdaptiveDt) {
            return this.config.dt;
        }
        
        // Get maximum velocity from field
        const maxVel = await this.getMaxVelocity();
        this.lastMaxVelocity = maxVel;
        
        const dx = this.gridConfig.resolution;
        
        // Physical speed of sound at 20°C: ~343 m/s
        // In lattice units: c_s = 1/sqrt(3) ≈ 0.577
        const c_s_physical = 343;  // m/s
        const c_s_lattice = this.CS;
        
        // Calculate current Mach number
        this.lastMachNumber = maxVel / c_s_physical;
        
        if (maxVel < 1e-6) {
            // Very low velocity, use maximum dt
            return this.config.dtMax;
        }
        
        // CFL constraint: dt ≤ CFL × dx / u_max
        const dt_cfl = this.config.cflFactor * dx / maxVel;
        
        // Mach constraint: Ma = u / c_s < Ma_max
        // In LBM, lattice velocity u_lat = u_phys × dt / dx
        // For Ma < 0.1: u_lat / c_s_lat < 0.1
        // => u_phys × dt / dx / (1/sqrt(3)) < 0.1
        // => dt < 0.1 × dx × sqrt(3) / u_phys
        const dt_mach = this.config.maxMach * dx * Math.sqrt(3) / maxVel;
        
        // Take minimum of constraints
        let newDt = Math.min(dt_cfl, dt_mach);
        
        // Apply bounds
        newDt = Math.max(this.config.dtMin, Math.min(newDt, this.config.dtMax));
        
        // Smooth transition (exponential moving average to avoid oscillations)
        const alpha = 0.3;  // Smoothing factor
        newDt = alpha * newDt + (1 - alpha) * this.currentDt;
        
        // Store history for analysis
        this.dtHistory.push(newDt);
        if (this.dtHistory.length > 100) {
            this.dtHistory.shift();
        }
        
        return newDt;
    }
    
    /**
     * Get maximum velocity in the field
     */
    async getMaxVelocity(): Promise<number> {
        const stagingBuffer = this.device.createBuffer({
            size: this.velocityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.velocityBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const velocityData = new Float32Array(stagingBuffer.getMappedRange());
        
        let maxVel = 0;
        const totalVoxels = velocityData.length / 3;
        
        // Sample every 4th voxel for performance
        for (let i = 0; i < totalVoxels; i += 4) {
            const vx = velocityData[i * 3];
            const vy = velocityData[i * 3 + 1];
            const vz = velocityData[i * 3 + 2];
            const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
            maxVel = Math.max(maxVel, vMag);
        }
        
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return maxVel;
    }
    
    /**
     * TRL 7: Get stability metrics for monitoring
     */
    async getStabilityMetrics(): Promise<StabilityMetrics> {
        const maxVel = this.lastMaxVelocity;
        const dx = this.gridConfig.resolution;
        const c_s_physical = 343;  // m/s
        
        const machNumber = maxVel / c_s_physical;
        const cflNumber = maxVel * this.currentDt / dx;
        
        // Stability margin: how far from critical values
        const machMargin = 1 - (machNumber / this.config.maxMach);
        const cflMargin = 1 - (cflNumber / 1.0);  // CFL < 1 for stability
        const stabilityMargin = Math.min(machMargin, cflMargin);
        
        const isStable = machNumber < this.config.maxMach && cflNumber < 1.0;
        
        return {
            currentDt: this.currentDt,
            maxVelocity: maxVel,
            machNumber,
            cflNumber,
            isStable,
            stabilityMargin: Math.max(0, stabilityMargin)
        };
    }
    
    /**
     * Update uniforms with new dt value
     */
    private updateDtUniform(dt: number): void {
        this.currentDt = dt;
        // dt is at offset 11 in uniform buffer (after gravity vec3)
        const dtData = new Float32Array([dt]);
        this.device.queue.writeBuffer(this.uniformBuffer, 11 * 4, dtData);
    }
    
    /**
     * Выполнить один шаг симуляции (collision + streaming + BC)
     * TRL 7: Now with adaptive time stepping
     */
    async step(): Promise<void> {
        if (!this.initialized) {
            throw new Error('[LBMSolver] Not initialized. Call initialize() first.');
        }
        
        // TRL 7: Adaptive time stepping
        if (this.config.enableAdaptiveDt && 
            this.currentStep % this.config.dtUpdateInterval === 0) {
            const newDt = await this.calculateAdaptiveDt();
            if (Math.abs(newDt - this.currentDt) / this.currentDt > 0.01) {
                this.updateDtUniform(newDt);
                console.log(`[LBMSolver] Adaptive dt: ${(newDt * 1000).toFixed(3)}ms (Ma=${this.lastMachNumber.toFixed(4)})`);
            }
        }
        
        const commandEncoder = this.device.createCommandEncoder({
            label: 'LBM Step Command Encoder',
        });
        
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const workgroupSize = 8;
        const dispatchX = Math.ceil(nx / workgroupSize);
        const dispatchY = Math.ceil(ny / workgroupSize);
        const dispatchZ = Math.ceil(nz / workgroupSize);
        
        // 1. Collision step
        const collisionPass = commandEncoder.beginComputePass({ label: 'Collision Pass' });
        collisionPass.setPipeline(this.collisionPipeline);
        collisionPass.setBindGroup(0, this.bindGroup);
        collisionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        collisionPass.end();
        
        // 2. Streaming step
        const streamingPass = commandEncoder.beginComputePass({ label: 'Streaming Pass' });
        streamingPass.setPipeline(this.streamingPipeline);
        streamingPass.setBindGroup(0, this.bindGroup);
        streamingPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        streamingPass.end();
        
        // 3. Boundary conditions
        const bcPass = commandEncoder.beginComputePass({ label: 'BC Pass' });
        bcPass.setPipeline(this.inletBCPipeline);
        bcPass.setBindGroup(0, this.bindGroup);
        bcPass.dispatchWorkgroups(Math.ceil(ny / workgroupSize), Math.ceil(nz / workgroupSize), 1);
        
        bcPass.setPipeline(this.outletBCPipeline);
        bcPass.setBindGroup(0, this.bindGroup);
        bcPass.dispatchWorkgroups(Math.ceil(ny / workgroupSize), Math.ceil(nz / workgroupSize), 1);
        bcPass.end();
        
        // 4. Swap buffers (f_out становится f_in)
        commandEncoder.copyBufferToBuffer(this.fOutBuffer, 0, this.fInBuffer, 0, this.fOutBuffer.size);
        
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        
        this.currentStep++;
    }
    
    /**
     * Get current time step value
     */
    getCurrentDt(): number {
        return this.currentDt;
    }
    
    /**
     * Get dt history for analysis
     */
    getDtHistory(): number[] {
        return [...this.dtHistory];
    }
    
    /**
     * Получить текущий snapshot симуляции для visualization
     */
    async getSnapshot(): Promise<SimulationSnapshot> {
        const { totalVoxels } = this.gridConfig;
        
        // Readback buffers
        const stagingDensity = this.device.createBuffer({
            size: this.densityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        const stagingVelocity = this.device.createBuffer({
            size: this.velocityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.densityBuffer, 0, stagingDensity, 0, stagingDensity.size);
        commandEncoder.copyBufferToBuffer(this.velocityBuffer, 0, stagingVelocity, 0, stagingVelocity.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingDensity.mapAsync(GPUMapMode.READ);
        await stagingVelocity.mapAsync(GPUMapMode.READ);
        
        const densityData = new Float32Array(stagingDensity.getMappedRange()).slice();
        const velocityData = new Float32Array(stagingVelocity.getMappedRange()).slice();
        
        stagingDensity.unmap();
        stagingVelocity.unmap();
        stagingDensity.destroy();
        stagingVelocity.destroy();
        
        // Calculate aggregated metrics
        let sumTemp = 0;
        let maxVel = 0;
        
        for (let i = 0; i < totalVoxels; i++) {
            sumTemp += densityData[i];
            const vx = velocityData[i * 3];
            const vy = velocityData[i * 3 + 1];
            const vz = velocityData[i * 3 + 2];
            const vmag = Math.sqrt(vx*vx + vy*vy + vz*vz);
            if (vmag > maxVel) maxVel = vmag;
        }
        
        // Calculate entropy metric
        const flowComplexity = this.calculateEntropyMetric(velocityData);
        
        return {
            timestamp: this.currentStep * this.config.dt,
            gridConfig: this.gridConfig,
            temperatureField: new Float32Array(0), // TODO: link to CHT
            velocityField: velocityData,
            comfortField: new Float32Array(0), // TODO: compute PMV
            metrics: {
                avgTemperature: sumTemp / totalVoxels,
                maxVelocity: maxVel,
                avgPMV: 0,
                co2Max: 0,
                energyConsumption: 0,
                flowComplexityIndex: flowComplexity,
            },
        };
    }

    /**
     * Calculate Flow Complexity Index based on velocity field entropy.
     * "Edge of Chaos" analysis.
     */
    private calculateEntropyMetric(velocityData: Float32Array): number {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        
        // 1. Calculate local variability (vorticity approximation)
        // We'll use a simplified approach: magnitude of curl (vorticity) or just gradient magnitude
        // For speed, let's use gradient of velocity magnitude
        
        let totalGradient = 0;
        let maxGradient = 0;
        const gradients: number[] = []; // Sampled gradients for entropy
        
        // Sampling step to avoid O(N) heavy calc if N is huge, but for GPU readback we already have the data
        // We'll sample every 2nd voxel to save CPU time
        const step = 2;
        
        for (let k = 1; k < nz - 1; k += step) {
            for (let j = 1; j < ny - 1; j += step) {
                for (let i = 1; i < nx - 1; i += step) {
                    const idx = i + j * nx + k * nx * ny;
                    
                    // Center velocity
                    const vx = velocityData[idx * 3];
                    const vy = velocityData[idx * 3 + 1];
                    const vz = velocityData[idx * 3 + 2];
                    const vMag = Math.sqrt(vx*vx + vy*vy + vz*vz);
                    
                    // Neighbors (just 6-connectivity for gradient)
                    // x+1
                    const idx_px = (i + 1) + j * nx + k * nx * ny;
                    const vMag_px = Math.sqrt(
                        velocityData[idx_px*3]**2 + velocityData[idx_px*3+1]**2 + velocityData[idx_px*3+2]**2
                    );
                    
                    // y+1
                    const idx_py = i + (j + 1) * nx + k * nx * ny;
                    const vMag_py = Math.sqrt(
                        velocityData[idx_py*3]**2 + velocityData[idx_py*3+1]**2 + velocityData[idx_py*3+2]**2
                    );

                    // z+1
                    const idx_pz = i + j * nx + (k + 1) * nx * ny;
                    const vMag_pz = Math.sqrt(
                        velocityData[idx_pz*3]**2 + velocityData[idx_pz*3+1]**2 + velocityData[idx_pz*3+2]**2
                    );
                    
                    // Gradient magnitude approx
                    const grad = Math.sqrt(
                        (vMag_px - vMag)**2 + (vMag_py - vMag)**2 + (vMag_pz - vMag)**2
                    );
                    
                    if (grad > 0.001) { // Ignore empty/still areas
                        gradients.push(grad);
                        maxGradient = Math.max(maxGradient, grad);
                    }
                }
            }
        }
        
        if (gradients.length === 0) return 0.0;
        
        // 2. Calculate Shannon Entropy of the gradient distribution
        // Histogram with 20 bins
        const bins = 20;
        const histogram = new Float32Array(bins);
        
        for (const grad of gradients) {
            const binIdx = Math.min(bins - 1, Math.floor((grad / maxGradient) * bins));
            histogram[binIdx]++;
        }
        
        // Normalize and compute entropy
        let entropy = 0;
        const totalSamples = gradients.length;
        
        for (let i = 0; i < bins; i++) {
            if (histogram[i] > 0) {
                const p = histogram[i] / totalSamples;
                entropy -= p * Math.log2(p);
            }
        }
        
        // Max possible entropy for 'bins' is log2(bins)
        const maxEntropy = Math.log2(bins); // ≈ 4.32 for 20 bins
        
        // Normalize index to 0-1
        const normalizedEntropy = entropy / maxEntropy;
        
        // 3. Interpret as "Life" metric
        // Too low (0.0-0.3) = Laminar/Dead
        // Too high (0.8-1.0) = Chaotic/Stress
        // Optimal (0.4-0.7) = "Edge of Chaos" (High score)
        
        // Let's return the raw entropy for now as the requested "FlowComplexityIndex"
        // The user asked for 0.0-1.0 complexity index.
        return normalizedEntropy;
    }
    
    /**
     * Get voxel state buffer for CHT coupling
     */
    getVoxelStateBuffer(): GPUBuffer {
        return this.voxelStateBuffer;
    }
    
    /**
     * Get velocity buffer for CHT coupling
     */
    getVelocityBuffer(): GPUBuffer {
        return this.velocityBuffer;
    }
    
    /**
     * Cleanup
     */
    destroy(): void {
        this.uniformBuffer?.destroy();
        this.fInBuffer?.destroy();
        this.fOutBuffer?.destroy();
        this.densityBuffer?.destroy();
        this.velocityBuffer?.destroy();
        this.temperatureBuffer?.destroy();
        this.voxelStateBuffer?.destroy();
        
        console.log('[LBMSolver] Destroyed');
    }
}
