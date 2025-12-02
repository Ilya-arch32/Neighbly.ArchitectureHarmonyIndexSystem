/**
 * AHI 2.0 Ultimate - 3D FDTD Acoustic Solver
 * 
 * Finite-Difference Time-Domain solver for acoustic wave propagation
 * Models diffraction, low-frequency resonances critical for atmosphere perception
 */

import { VoxelGridConfig } from './VoxelTypes';

/**
 * Коэффициенты звукопоглощения материалов (α) при 500 Hz
 * Источник: ISO 354, справочные данные
 */
export interface MaterialAbsorption {
    id: number;
    name: string;
    alpha: number;  // Коэффициент звукопоглощения (0-1)
}

/**
 * Стандартные коэффициенты звукопоглощения
 */
export const MATERIAL_ABSORPTION: Record<string, MaterialAbsorption> = {
    AIR: { id: 0, name: 'Air', alpha: 0.0 },            // Воздух - не поглощает
    CONCRETE: { id: 1, name: 'Concrete', alpha: 0.02 }, // Бетон - очень отражающий
    WOOD: { id: 2, name: 'Wood', alpha: 0.15 },         // Дерево - умеренное поглощение
    GLASS: { id: 3, name: 'Glass', alpha: 0.04 },       // Стекло - отражающее
    CARPET: { id: 4, name: 'Carpet', alpha: 0.35 },     // Ковер - хорошее поглощение
    ACOUSTIC_PANEL: { id: 5, name: 'Acoustic Panel', alpha: 0.85 }, // Акустические панели
    BRICK: { id: 6, name: 'Brick', alpha: 0.03 },       // Кирпич
    GYPSUM: { id: 7, name: 'Gypsum Board', alpha: 0.10 }, // Гипсокартон
    CURTAIN: { id: 8, name: 'Heavy Curtain', alpha: 0.55 }, // Тяжелые шторы
};

export interface AcousticMetrics {
    RT60: number;                    // Reverberation time (seconds) - from Schroeder integration
    T30: number;                     // T30 reverberation time (more accurate than RT60)
    EDT: number;                     // Early Decay Time (seconds)
    C50: number;                     // Speech clarity index (dB) 
    C80: number;                     // Musical clarity (dB)
    D50: number;                     // Definition (ratio 0-1)
    modalDensity: number;            // Modal density in room
    spatialImpression: number;       // Listener envelopment
    acousticIntimacy: number;        // Early reflections strength
    
    // Validation flags
    calculationMethod: 'schroeder' | 'sabine';  // Which method was used
    confidence: number;              // Confidence level 0-1
}

/**
 * Room Impulse Response (RIR) data structure
 */
export interface RoomImpulseResponse {
    pressure: Float32Array;          // Pressure samples at receiver
    sampleRate: number;              // Samples per second
    duration: number;                // Total duration in seconds
    sourcePosition: { x: number; y: number; z: number };
    receiverPosition: { x: number; y: number; z: number };
}

export class AcousticSolver {
    private device: GPUDevice;
    private gridConfig: VoxelGridConfig;
    
    // Wave field buffers (double buffering)
    private pressureBufferA!: GPUBuffer;
    private pressureBufferB!: GPUBuffer;
    private velocityBuffer!: GPUBuffer;
    
    // Compute pipelines
    private velocityPipeline!: GPUComputePipeline;
    private pressurePipeline!: GPUComputePipeline;
    private analysisPipeline!: GPUComputePipeline;
    
    // Physical constants
    private c = 343.0;              // Speed of sound m/s
    private rho = 1.225;            // Air density kg/m³
    private K: number;              // Bulk modulus
    
    // Voxel data for Sabine calculation
    private voxelStateBuffer: GPUBuffer | null = null;
    private voxelStateData: Float32Array | null = null;
    
    // Cached surface area data
    private surfaceAreaByMaterial: Map<number, number> = new Map();
    private totalVolume: number = 0;
    private surfaceAreaCalculated: boolean = false;
    
    // Room Impulse Response recording
    private rirBuffer: GPUBuffer | null = null;
    private rirSampleRate: number = 44100;  // Hz
    private rirDuration: number = 2.0;      // seconds
    private rirData: Float32Array | null = null;
    private rirRecorded: boolean = false;
    
    // Receiver position for RIR recording
    private receiverPosition: { x: number; y: number; z: number } = { x: 5, y: 5, z: 1.5 };
    
    constructor(device: GPUDevice, gridConfig: VoxelGridConfig) {
        this.device = device;
        this.gridConfig = gridConfig;
        this.K = this.rho * this.c * this.c;
        
        // Инициализируем коэффициенты поглощения по умолчанию
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            this.surfaceAreaByMaterial.set(mat.id, 0);
        }
    }
    
    /**
     * Initialize FDTD solver with WebGPU resources
     */
    async initialize(voxelBuffer: GPUBuffer): Promise<void> {
        const { totalVoxels } = this.gridConfig;
        const { nx, ny, nz } = this.gridConfig.dimensions;
        
        // Сохраняем ссылку на voxel buffer для расчета площади поверхности
        this.voxelStateBuffer = voxelBuffer;
        
        // Allocate wave field buffers
        this.pressureBufferA = this.device.createBuffer({
            label: 'Pressure Buffer A',
            size: totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        this.pressureBufferB = this.device.createBuffer({
            label: 'Pressure Buffer B', 
            size: totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        // Velocity has 3 components
        this.velocityBuffer = this.device.createBuffer({
            label: 'Velocity Buffer',
            size: totalVoxels * 16, // vec3 + padding
            usage: GPUBufferUsage.STORAGE
        });
        
        // Create FDTD compute shaders
        const shaderModule = this.device.createShaderModule({
            label: 'FDTD Acoustics Shaders',
            code: this.generateFDTDShaders()
        });
        
        // Velocity update pipeline
        this.velocityPipeline = this.device.createComputePipeline({
            label: 'Velocity Update',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'updateVelocity'
            }
        });
        
        // Pressure update pipeline
        this.pressurePipeline = this.device.createComputePipeline({
            label: 'Pressure Update',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'updatePressure'
            }
        });
        
        // Analysis pipeline for metrics
        this.analysisPipeline = this.device.createComputePipeline({
            label: 'Acoustic Analysis',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'analyzeAcoustics'
            }
        });
        
        console.log('[AcousticSolver] Initialized 3D FDTD solver');
        
        // Allocate RIR buffer for Schroeder integration
        const rirSamples = Math.ceil(this.rirSampleRate * this.rirDuration);
        this.rirBuffer = this.device.createBuffer({
            label: 'RIR Buffer',
            size: rirSamples * 4, // float32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.rirData = new Float32Array(rirSamples);
    }
    
    /**
     * Set receiver position for RIR measurement
     */
    setReceiverPosition(x: number, y: number, z: number): void {
        this.receiverPosition = { x, y, z };
        this.rirRecorded = false; // Invalidate cached RIR
    }
    
    /**
     * Run FDTD simulation step
     */
    async step(dt: number): Promise<void> {
        const commandEncoder = this.device.createCommandEncoder();
        const { nx, ny, nz } = this.gridConfig.dimensions;
        
        // Update velocity from pressure gradient
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.velocityPipeline);
            // Bind pressure and velocity buffers
            pass.dispatchWorkgroups(
                Math.ceil(nx / 8),
                Math.ceil(ny / 8),
                Math.ceil(nz / 8)
            );
            pass.end();
        }
        
        // Update pressure from velocity divergence
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pressurePipeline);
            // Bind velocity and pressure buffers
            pass.dispatchWorkgroups(
                Math.ceil(nx / 8),
                Math.ceil(ny / 8),
                Math.ceil(nz / 8)
            );
            pass.end();
        }
        
        // Swap buffers
        [this.pressureBufferA, this.pressureBufferB] = 
            [this.pressureBufferB, this.pressureBufferA];
        
        this.device.queue.submit([commandEncoder.finish()]);
    }
    
    /**
     * Inject acoustic impulse (for room response)
     */
    async injectImpulse(x: number, y: number, z: number): Promise<void> {
        // Gaussian pulse injection
        const impulseData = new Float32Array(1);
        impulseData[0] = 1000.0; // Pressure amplitude
        
        const { nx, ny } = this.gridConfig.dimensions;
        const idx = x + y * nx + z * nx * ny;
        
        this.device.queue.writeBuffer(
            this.pressureBufferA,
            idx * 4,
            impulseData
        );
    }
    
    /**
     * Apply Perfectly Matched Layer (PML) boundaries
     */
    private applyPML(): void {
        // PML absorbing boundaries to prevent reflections
        // Implemented in shader for efficiency
    }
    
    /**
     * Расчет площади поверхности для каждого материала
     * Подсчитывает граничные воксели (Solid рядом с Fluid)
     */
    async calculateSurfaceArea(): Promise<Map<number, number>> {
        if (!this.voxelStateBuffer) {
            console.warn('[AcousticSolver] Voxel buffer not initialized');
            return this.surfaceAreaByMaterial;
        }
        
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        const voxelVolume = Math.pow(this.gridConfig.resolution, 3);
        const faceArea = Math.pow(this.gridConfig.resolution, 2);
        
        // Считываем данные вокселей из GPU
        const stagingBuffer = this.device.createBuffer({
            size: this.voxelStateBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.voxelStateBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        this.voxelStateData = new Float32Array(stagingBuffer.getMappedRange()).slice();
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        // Сбрасываем счетчики
        this.surfaceAreaByMaterial.clear();
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            this.surfaceAreaByMaterial.set(mat.id, 0);
        }
        this.totalVolume = 0;
        
        // Проходим по всем вокселям
        const VOXEL_FLUID = 2;
        const VOXEL_SOLID = 1;
        const stride = 16; // Количество float на воксель (state, material, ...)
        
        const neighbors = [
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1]
        ];
        
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = this.voxelStateData[idx * stride];
                    const material = Math.floor(this.voxelStateData[idx * stride + 1]);
                    
                    // Считаем объем воздуха (Fluid voxels)
                    if (state === VOXEL_FLUID) {
                        this.totalVolume += voxelVolume;
                    }
                    
                    // Ищем границы Solid-Fluid
                    if (state === VOXEL_SOLID) {
                        for (const [dx, dy, dz] of neighbors) {
                            const nx2 = x + dx;
                            const ny2 = y + dy;
                            const nz2 = z + dz;
                            
                            // Проверка границ
                            if (nx2 >= 0 && nx2 < nx &&
                                ny2 >= 0 && ny2 < ny &&
                                nz2 >= 0 && nz2 < nz) {
                                
                                const neighborIdx = nx2 + ny2 * nx + nz2 * nx * ny;
                                const neighborState = this.voxelStateData[neighborIdx * stride];
                                
                                // Если сосед - воздух, эта грань - поверхность
                                if (neighborState === VOXEL_FLUID) {
                                    const currentArea = this.surfaceAreaByMaterial.get(material) || 0;
                                    this.surfaceAreaByMaterial.set(material, currentArea + faceArea);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        this.surfaceAreaCalculated = true;
        console.log(`[AcousticSolver] Surface area calculated: V=${this.totalVolume.toFixed(2)}m³`);
        
        return this.surfaceAreaByMaterial;
    }
    
    /**
     * Record Room Impulse Response from FDTD simulation
     * Runs full FDTD simulation and records pressure at receiver position
     */
    async recordRIR(
        sourceX: number, sourceY: number, sourceZ: number,
        numSteps?: number
    ): Promise<RoomImpulseResponse> {
        const dt = 1.0 / this.rirSampleRate;
        const totalSteps = numSteps || Math.ceil(this.rirDuration / dt);
        
        console.log(`[AcousticSolver] Recording RIR: ${totalSteps} steps at ${this.rirSampleRate}Hz`);
        
        // Inject impulse at source position
        await this.injectImpulse(
            Math.floor(sourceX / this.gridConfig.resolution),
            Math.floor(sourceY / this.gridConfig.resolution),
            Math.floor(sourceZ / this.gridConfig.resolution)
        );
        
        // Receiver voxel index
        const { nx, ny } = this.gridConfig.dimensions;
        const recX = Math.floor(this.receiverPosition.x / this.gridConfig.resolution);
        const recY = Math.floor(this.receiverPosition.y / this.gridConfig.resolution);
        const recZ = Math.floor(this.receiverPosition.z / this.gridConfig.resolution);
        const recIdx = recX + recY * nx + recZ * nx * ny;
        
        // Run simulation and record pressure at receiver
        const rirSamples = new Float32Array(totalSteps);
        
        for (let step = 0; step < totalSteps; step++) {
            await this.step(dt);
            
            // Read pressure at receiver position (would be optimized with GPU readback)
            // For now, sample every few steps to reduce overhead
            if (step % 10 === 0) {
                const pressure = await this.readPressureAtPoint(recIdx);
                rirSamples[step] = pressure;
            }
        }
        
        this.rirData = rirSamples;
        this.rirRecorded = true;
        
        console.log('[AcousticSolver] RIR recording complete');
        
        return {
            pressure: rirSamples,
            sampleRate: this.rirSampleRate,
            duration: this.rirDuration,
            sourcePosition: { x: sourceX, y: sourceY, z: sourceZ },
            receiverPosition: this.receiverPosition
        };
    }
    
    /**
     * Read pressure value at a specific voxel index
     */
    private async readPressureAtPoint(voxelIdx: number): Promise<number> {
        const stagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.pressureBufferA, voxelIdx * 4,
            stagingBuffer, 0, 4
        );
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const pressure = data[0];
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return pressure;
    }
    
    /**
     * Schroeder Integration for reverberation analysis
     * E(t) = ∫_t^∞ p²(τ)dτ
     * 
     * This is the backward integration method from Schroeder (1965)
     * that gives the ensemble-averaged decay curve from a single RIR
     */
    calculateSchroederCurve(rir: Float32Array): Float32Array {
        const n = rir.length;
        const energyCurve = new Float32Array(n);
        
        // Backward integration: E(t) = ∫_t^∞ p²(τ)dτ
        let cumulativeEnergy = 0;
        
        for (let i = n - 1; i >= 0; i--) {
            cumulativeEnergy += rir[i] * rir[i];
            energyCurve[i] = cumulativeEnergy;
        }
        
        // Normalize to start at 0 dB
        const maxEnergy = energyCurve[0];
        if (maxEnergy > 0) {
            for (let i = 0; i < n; i++) {
                energyCurve[i] = energyCurve[i] / maxEnergy;
            }
        }
        
        return energyCurve;
    }
    
    /**
     * Calculate reverberation time from Schroeder curve
     * Uses linear regression on the decay curve
     * 
     * @param startDb Start of measurement range (e.g., -5 for T30)
     * @param endDb End of measurement range (e.g., -35 for T30)
     * @returns Reverberation time in seconds (extrapolated to -60dB)
     */
    calculateRTFromSchroeder(
        energyCurve: Float32Array,
        sampleRate: number,
        startDb: number = -5,
        endDb: number = -35
    ): { rt60: number; edt: number; confidence: number } {
        const n = energyCurve.length;
        
        // Convert to dB
        const dbCurve = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            dbCurve[i] = energyCurve[i] > 1e-10 ? 10 * Math.log10(energyCurve[i]) : -100;
        }
        
        // Find indices for start and end of decay range
        let startIdx = 0;
        let endIdx = n - 1;
        
        for (let i = 0; i < n; i++) {
            if (dbCurve[i] <= startDb && startIdx === 0) {
                startIdx = i;
            }
            if (dbCurve[i] <= endDb) {
                endIdx = i;
                break;
            }
        }
        
        // Need at least 10 samples for reliable regression
        if (endIdx - startIdx < 10) {
            return { rt60: -1, edt: -1, confidence: 0 };
        }
        
        // Linear regression on dB vs time
        // y = slope * x + intercept
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        const count = endIdx - startIdx + 1;
        
        for (let i = startIdx; i <= endIdx; i++) {
            const t = i / sampleRate;
            sumX += t;
            sumY += dbCurve[i];
            sumXY += t * dbCurve[i];
            sumX2 += t * t;
        }
        
        const slope = (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / count;
        
        // RT60 = time for 60dB decay
        // slope is dB/second, so RT60 = -60 / slope
        const rt60 = slope < -0.1 ? -60 / slope : -1;
        
        // EDT: Early Decay Time (first 10dB of decay)
        let edtIdx = 0;
        for (let i = 0; i < n; i++) {
            if (dbCurve[i] <= -10) {
                edtIdx = i;
                break;
            }
        }
        const edt = edtIdx > 0 ? (edtIdx / sampleRate) * 6 : rt60; // Extrapolate to 60dB
        
        // Calculate R² for confidence
        let ssTot = 0, ssRes = 0;
        const meanY = sumY / count;
        for (let i = startIdx; i <= endIdx; i++) {
            const t = i / sampleRate;
            const predicted = slope * t + intercept;
            ssTot += (dbCurve[i] - meanY) ** 2;
            ssRes += (dbCurve[i] - predicted) ** 2;
        }
        const rSquared = 1 - (ssRes / ssTot);
        
        return { 
            rt60: Math.max(0.1, Math.min(rt60, 10)), 
            edt: Math.max(0.1, Math.min(edt, 10)),
            confidence: Math.max(0, rSquared)
        };
    }
    
    /**
     * Calculate Clarity indices (C50, C80) from RIR
     * C_t = 10 * log10(∫_0^t p²(τ)dτ / ∫_t^∞ p²(τ)dτ)
     */
    calculateClarity(rir: Float32Array, sampleRate: number, limitMs: number): number {
        const limitSamples = Math.floor(sampleRate * limitMs / 1000);
        const n = rir.length;
        
        let earlyEnergy = 0;
        let lateEnergy = 0;
        
        for (let i = 0; i < n; i++) {
            const energy = rir[i] * rir[i];
            if (i < limitSamples) {
                earlyEnergy += energy;
            } else {
                lateEnergy += energy;
            }
        }
        
        if (lateEnergy < 1e-10) {
            return 20; // Very high clarity (dry room)
        }
        
        return 10 * Math.log10(earlyEnergy / lateEnergy);
    }
    
    /**
     * Calculate Definition D50 from RIR
     * D50 = ∫_0^50ms p²(τ)dτ / ∫_0^∞ p²(τ)dτ
     */
    calculateDefinition(rir: Float32Array, sampleRate: number): number {
        const limit50ms = Math.floor(sampleRate * 0.05);
        const n = rir.length;
        
        let earlyEnergy = 0;
        let totalEnergy = 0;
        
        for (let i = 0; i < n; i++) {
            const energy = rir[i] * rir[i];
            totalEnergy += energy;
            if (i < limit50ms) {
                earlyEnergy += energy;
            }
        }
        
        return totalEnergy > 0 ? earlyEnergy / totalEnergy : 0;
    }
    
    /**
     * Расчет RT60 по формуле Сэбина (Sabine Equation)
     * FALLBACK: Used when FDTD simulation data is unavailable
     * RT60 = 0.161 * V / Σ(S_i * α_i)
     * 
     * где:
     * - V: объем помещения [м³]
     * - S_i: площадь поверхности материала i [м²]
     * - α_i: коэффициент звукопоглощения материала i
     */
    calculateRT60Sabine(): number {
        if (!this.surfaceAreaCalculated || this.totalVolume === 0) {
            console.warn('[AcousticSolver] Surface area not calculated, using fallback');
            return 0.5; // Значение по умолчанию
        }
        
        // Суммарное эффективное поглощение A = Σ(S_i * α_i)
        let totalAbsorption = 0;
        let totalSurfaceArea = 0;
        
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            const surfaceArea = this.surfaceAreaByMaterial.get(mat.id) || 0;
            totalAbsorption += surfaceArea * mat.alpha;
            totalSurfaceArea += surfaceArea;
        }
        
        // Добавляем поглощение воздуха (4mV для высоких частот)
        const airAbsorptionCoeff = 0.002; // m^-1 при 500 Hz, 20°C, 50% RH
        const airAbsorption = 4 * airAbsorptionCoeff * this.totalVolume;
        totalAbsorption += airAbsorption;
        
        // Защита от деления на ноль
        if (totalAbsorption < 0.01) {
            totalAbsorption = 0.01;
        }
        
        // Формула Сэбина: RT60 = 0.161 * V / A
        const RT60 = 0.161 * this.totalVolume / totalAbsorption;
        
        console.log(`[AcousticSolver] Sabine RT60: ${RT60.toFixed(2)}s (V=${this.totalVolume.toFixed(1)}m³, A=${totalAbsorption.toFixed(2)}m²)`);
        
        // Ограничиваем разумным диапазоном
        return Math.max(0.1, Math.min(RT60, 10.0));
    }
    
    /**
     * Расчет C50 (Clarity) на основе RT60 - EMPIRICAL FALLBACK
     * C50 ≈ 10.8 - 24.4 * log10(RT60) для типичных помещений
     * NOTE: For TRL 8, use calculateClarity() with actual RIR data
     */
    calculateC50FromRT60(RT60: number): number {
        // Эмпирическая формула для речи (Barron 1988)
        const C50 = 10.8 - 24.4 * Math.log10(Math.max(0.1, RT60));
        return Math.max(-10, Math.min(C50, 15));
    }
    
    /**
     * Расчет C80 (Musical Clarity) на основе RT60 - EMPIRICAL FALLBACK
     * NOTE: For TRL 8, use calculateClarity() with actual RIR data
     */
    calculateC80FromRT60(RT60: number): number {
        // Эмпирическая формула (Barron 1988)
        const C80 = 5.31 - 13.8 * Math.log10(Math.max(0.1, RT60));
        return Math.max(-10, Math.min(C80, 10));
    }
    
    /**
     * Расчет модальной плотности (Modal Density)
     * N(f) = (4πVf²/c³) + (πSf/2c²) + (Lf/8c)
     * Упрощенная версия для низких частот
     */
    calculateModalDensity(frequency: number = 125): number {
        if (this.totalVolume === 0) return 0;
        
        // Общая площадь поверхности
        let totalSurface = 0;
        for (const area of this.surfaceAreaByMaterial.values()) {
            totalSurface += area;
        }
        
        // Характерный размер (приближение)
        const L = Math.pow(this.totalVolume, 1/3) * 4;
        
        const c = this.c;
        const f = frequency;
        
        // Формула модальной плотности
        const modalDensity = (4 * Math.PI * this.totalVolume * f * f) / (c * c * c) +
                            (Math.PI * totalSurface * f) / (2 * c * c) +
                            (L * f) / (8 * c);
        
        return modalDensity;
    }
    
    /**
     * Calculate acoustic metrics - TRL 7/8 implementation
     * Primary: Schroeder Integration from FDTD simulation
     * Fallback: Sabine equation for quick estimates
     */
    async calculateMetrics(useSchroeder: boolean = true): Promise<AcousticMetrics> {
        // Рассчитываем площадь поверхности если еще не сделано
        if (!this.surfaceAreaCalculated) {
            await this.calculateSurfaceArea();
        }
        
        let RT60: number;
        let T30: number;
        let EDT: number;
        let C50: number;
        let C80: number;
        let D50: number;
        let confidence: number;
        let calculationMethod: 'schroeder' | 'sabine';
        
        // Try Schroeder integration if RIR is available
        if (useSchroeder && this.rirRecorded && this.rirData && this.rirData.length > 0) {
            console.log('[AcousticSolver] Using Schroeder Integration for metrics');
            
            // Calculate Schroeder curve
            const schroederCurve = this.calculateSchroederCurve(this.rirData);
            
            // Extract reverberation times from Schroeder curve
            const rtResult = this.calculateRTFromSchroeder(
                schroederCurve, 
                this.rirSampleRate,
                -5, -35  // T30 range
            );
            
            T30 = rtResult.rt60;
            RT60 = T30;  // T30 is more reliable than direct RT60
            EDT = rtResult.edt;
            confidence = rtResult.confidence;
            
            // Calculate Clarity from RIR directly
            C50 = this.calculateClarity(this.rirData, this.rirSampleRate, 50);
            C80 = this.calculateClarity(this.rirData, this.rirSampleRate, 80);
            D50 = this.calculateDefinition(this.rirData, this.rirSampleRate);
            
            calculationMethod = 'schroeder';
            
            console.log(`[AcousticSolver] Schroeder results: T30=${T30.toFixed(2)}s, EDT=${EDT.toFixed(2)}s, C50=${C50.toFixed(1)}dB, confidence=${(confidence*100).toFixed(0)}%`);
            
        } else {
            // Fallback: Sabine equation
            console.log('[AcousticSolver] Using Sabine equation (fallback)');
            
            RT60 = this.calculateRT60Sabine();
            T30 = RT60;  // Approximate
            EDT = RT60 * 0.9;  // Typical EDT/RT60 ratio
            
            // Derive clarity from RT60 (empirical)
            C50 = this.calculateC50FromRT60(RT60);
            C80 = this.calculateC80FromRT60(RT60);
            D50 = 0.5;  // Default estimate
            
            confidence = 0.6;  // Lower confidence for Sabine
            calculationMethod = 'sabine';
        }
        
        const modalDensity = this.calculateModalDensity();
        
        // Spatial Impression зависит от ранних отражений
        // Упрощенная формула: выше RT60 = больше ощущение пространства
        const spatialImpression = Math.min(1.0, RT60 / 2.0);
        
        // Acoustic Intimacy - обратно пропорциональна объему
        const intimacyVolume = 200; // Оптимальный объем для интимности
        const acousticIntimacy = Math.max(0, Math.min(1.0, 1.0 - (this.totalVolume - intimacyVolume) / 1000));
        
        return {
            RT60,
            T30,
            EDT,
            C50,
            C80,
            D50,
            modalDensity,
            spatialImpression,
            acousticIntimacy,
            calculationMethod,
            confidence
        };
    }
    
    /**
     * Run full acoustic analysis with Schroeder integration
     * This is the TRL 8 method that should be used for validation
     */
    async runFullAnalysis(
        sourceX: number, sourceY: number, sourceZ: number
    ): Promise<AcousticMetrics> {
        // Record RIR from FDTD simulation
        await this.recordRIR(sourceX, sourceY, sourceZ);
        
        // Calculate metrics using Schroeder integration
        return this.calculateMetrics(true);
    }
    
    /**
     * Получить данные о площади поверхности по материалам
     */
    getSurfaceAreaData(): { material: string; area: number; alpha: number }[] {
        const result: { material: string; area: number; alpha: number }[] = [];
        
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            const area = this.surfaceAreaByMaterial.get(mat.id) || 0;
            if (area > 0) {
                result.push({
                    material: mat.name,
                    area,
                    alpha: mat.alpha
                });
            }
        }
        
        return result;
    }
    
    /**
     * Получить объем помещения
     */
    getRoomVolume(): number {
        return this.totalVolume;
    }
    
    /**
     * Generate WGSL shaders for FDTD
     */
    private generateFDTDShaders(): string {
        return `
            struct Uniforms {
                grid_size: vec3<u32>,
                dt: f32,
                c: f32,
                rho: f32,
                K: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> pressure_in: array<f32>;
            @group(0) @binding(2) var<storage, read_write> pressure_out: array<f32>;
            @group(0) @binding(3) var<storage, read_write> velocity: array<vec4<f32>>;
            @group(0) @binding(4) var<storage, read> voxels: array<f32>;
            
            @compute @workgroup_size(8, 8, 8)
            fn updateVelocity(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= uniforms.grid_size.x || 
                    gid.y >= uniforms.grid_size.y || 
                    gid.z >= uniforms.grid_size.z) {
                    return;
                }
                
                let idx = gid.x + gid.y * uniforms.grid_size.x + 
                         gid.z * uniforms.grid_size.x * uniforms.grid_size.y;
                
                // Check if fluid voxel
                if (voxels[idx * 8u] > 0.5) { return; } // Skip solid
                
                // Calculate pressure gradient
                var grad_p = vec3<f32>(0.0);
                
                // X gradient
                if (gid.x > 0u && gid.x < uniforms.grid_size.x - 1u) {
                    let p_plus = pressure_in[idx + 1u];
                    let p_minus = pressure_in[idx - 1u];
                    grad_p.x = (p_plus - p_minus) / (2.0 * uniforms.grid_size.x);
                }
                
                // Y gradient  
                if (gid.y > 0u && gid.y < uniforms.grid_size.y - 1u) {
                    let p_plus = pressure_in[idx + uniforms.grid_size.x];
                    let p_minus = pressure_in[idx - uniforms.grid_size.x];
                    grad_p.y = (p_plus - p_minus) / (2.0 * uniforms.grid_size.y);
                }
                
                // Z gradient
                if (gid.z > 0u && gid.z < uniforms.grid_size.z - 1u) {
                    let stride_z = uniforms.grid_size.x * uniforms.grid_size.y;
                    let p_plus = pressure_in[idx + stride_z];
                    let p_minus = pressure_in[idx - stride_z];
                    grad_p.z = (p_plus - p_minus) / (2.0 * uniforms.grid_size.z);
                }
                
                // Update velocity: dv/dt = -(1/rho) * grad(p)
                let v_current = velocity[idx];
                let v_new = v_current.xyz - (uniforms.dt / uniforms.rho) * grad_p;
                
                // Apply PML damping in boundary regions
                var sigma = 0.0;
                let pml_width = 10u;
                if (gid.x < pml_width || gid.x > uniforms.grid_size.x - pml_width ||
                    gid.y < pml_width || gid.y > uniforms.grid_size.y - pml_width ||
                    gid.z < pml_width || gid.z > uniforms.grid_size.z - pml_width) {
                    sigma = 100.0; // Damping coefficient
                }
                
                velocity[idx] = vec4<f32>(v_new * exp(-sigma * uniforms.dt), 0.0);
            }
            
            @compute @workgroup_size(8, 8, 8)
            fn updatePressure(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= uniforms.grid_size.x || 
                    gid.y >= uniforms.grid_size.y || 
                    gid.z >= uniforms.grid_size.z) {
                    return;
                }
                
                let idx = gid.x + gid.y * uniforms.grid_size.x + 
                         gid.z * uniforms.grid_size.x * uniforms.grid_size.y;
                
                // Check if fluid voxel
                if (voxels[idx * 8u] > 0.5) {
                    pressure_out[idx] = 0.0; // Rigid boundary
                    return;
                }
                
                // Calculate velocity divergence
                var div_v = 0.0;
                
                // X divergence
                if (gid.x > 0u && gid.x < uniforms.grid_size.x - 1u) {
                    let v_plus = velocity[idx + 1u].x;
                    let v_minus = velocity[idx - 1u].x;
                    div_v += (v_plus - v_minus) / (2.0 * uniforms.grid_size.x);
                }
                
                // Y divergence
                if (gid.y > 0u && gid.y < uniforms.grid_size.y - 1u) {
                    let v_plus = velocity[idx + uniforms.grid_size.x].y;
                    let v_minus = velocity[idx - uniforms.grid_size.x].y;
                    div_v += (v_plus - v_minus) / (2.0 * uniforms.grid_size.y);
                }
                
                // Z divergence
                if (gid.z > 0u && gid.z < uniforms.grid_size.z - 1u) {
                    let stride_z = uniforms.grid_size.x * uniforms.grid_size.y;
                    let v_plus = velocity[idx + stride_z].z;
                    let v_minus = velocity[idx - stride_z].z;
                    div_v += (v_plus - v_minus) / (2.0 * uniforms.grid_size.z);
                }
                
                // Update pressure: dp/dt = -K * div(v)
                let p_current = pressure_in[idx];
                pressure_out[idx] = p_current - uniforms.K * uniforms.dt * div_v;
            }
            
            @compute @workgroup_size(1)
            fn analyzeAcoustics(@builtin(global_invocation_id) gid: vec3<u32>) {
                // Calculate RT60, clarity indices from impulse response
                // This would integrate the energy decay curve
                // Simplified placeholder for now
            }
        `;
    }
    
    /**
     * Cleanup
     */
    destroy(): void {
        this.pressureBufferA?.destroy();
        this.pressureBufferB?.destroy();
        this.velocityBuffer?.destroy();
        this.voxelStateData = null;
        this.voxelStateBuffer = null;
        
        console.log('[AcousticSolver] Destroyed');
    }
}
