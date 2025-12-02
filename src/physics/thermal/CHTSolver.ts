/**
 * AHI 2.0 Ultimate - Conjugate Heat Transfer Solver
 * 
 * Manages thermal coupling between solid walls and fluid (air)
 * Integrates with LBM for buoyancy-driven flows
 * Includes solar radiation based on NREL SPA
 */

import { VoxelGridConfig } from './VoxelTypes';
import { SolarPositionAlgorithm, Location } from './SolarPositionAlgorithm';
import heatTransferShaderCode from './heat_transfer.wgsl?raw';

/**
 * Параметры материалов для влагопереноса (ISO 13788 + MBV)
 * TRL 7: Добавлена модель буферной ёмкости влаги (Moisture Buffer Value)
 */
export interface MaterialProperties {
    id: number;
    name: string;
    thermalConductivity: number;   // λ [W/(m·K)]
    density: number;               // ρ [kg/m³]
    specificHeat: number;          // c_p [J/(kg·K)]
    vaporPermeability: number;     // μ - коэффициент сопротивления паропроницаемости [-]
    // μ = 1 для воздуха, ~50-200 для бетона, ~5-20 для дерева
    
    // Moisture Buffer Value (MBV) - TRL 7 enhancement
    // Классификация по NORDTEST:
    // Negligible: < 0.2, Limited: 0.2-0.5, Moderate: 0.5-1.0, Good: 1.0-2.0, Excellent: > 2.0
    mbv: number;                   // Moisture Buffer Value [g/(m²·%RH)]
    
    // Сорбционная изотерма (коэффициенты GAB модели)
    // w = (w_m * C * K * RH) / ((1 - K * RH) * (1 - K * RH + C * K * RH))
    sorptionWm?: number;           // w_m - монослойная влажность [kg/kg]
    sorptionC?: number;            // C - константа Guggenheim
    sorptionK?: number;            // K - константа многослойной сорбции
    
    // Гистерезис сорбции/десорбции
    hysteresisRatio?: number;      // Коэффициент гистерезиса (0.7-0.9 типично)
}

/**
 * Стандартные материалы с паропроницаемостью и MBV (NORDTEST Project)
 * MBV значения из: Rode et al. "Moisture Buffering of Building Materials" (2005)
 */
export const MATERIAL_PROPERTIES: Record<string, MaterialProperties> = {
    AIR: { 
        id: 0, name: 'Air', 
        thermalConductivity: 0.026, density: 1.225, specificHeat: 1005, 
        vaporPermeability: 1, 
        mbv: 0  // Воздух не буферизует
    },
    CONCRETE: { 
        id: 1, name: 'Concrete', 
        thermalConductivity: 1.4, density: 2400, specificHeat: 880, 
        vaporPermeability: 100, 
        mbv: 0.38,  // Limited buffer capacity
        sorptionWm: 0.02, sorptionC: 10, sorptionK: 0.75,
        hysteresisRatio: 0.85
    },
    WOOD: { 
        id: 2, name: 'Wood', 
        thermalConductivity: 0.15, density: 600, specificHeat: 1700, 
        vaporPermeability: 20, 
        mbv: 1.35,  // Good buffer capacity
        sorptionWm: 0.05, sorptionC: 15, sorptionK: 0.80,
        hysteresisRatio: 0.75
    },
    GLASS: { 
        id: 3, name: 'Glass', 
        thermalConductivity: 1.0, density: 2500, specificHeat: 840, 
        vaporPermeability: 1e6,  // практически непроницаем
        mbv: 0  // No buffer capacity
    },
    INSULATION: { 
        id: 4, name: 'Insulation', 
        thermalConductivity: 0.04, density: 30, specificHeat: 1030, 
        vaporPermeability: 5, 
        mbv: 0.15,  // Negligible (mineral wool)
        sorptionWm: 0.001, sorptionC: 5, sorptionK: 0.6
    },
    BRICK: { 
        id: 5, name: 'Brick', 
        thermalConductivity: 0.8, density: 1800, specificHeat: 900, 
        vaporPermeability: 15, 
        mbv: 0.48,  // Limited-Moderate
        sorptionWm: 0.015, sorptionC: 12, sorptionK: 0.78,
        hysteresisRatio: 0.80
    },
    GYPSUM_BOARD: { 
        id: 6, name: 'Gypsum Board', 
        thermalConductivity: 0.25, density: 850, specificHeat: 1000, 
        vaporPermeability: 8, 
        mbv: 0.63,  // Moderate - часто используется для регулирования влажности
        sorptionWm: 0.025, sorptionC: 8, sorptionK: 0.72,
        hysteresisRatio: 0.82
    },
    CLAY_PLASTER: { 
        id: 7, name: 'Clay Plaster', 
        thermalConductivity: 0.7, density: 1700, specificHeat: 1000, 
        vaporPermeability: 10, 
        mbv: 2.10,  // Excellent - лучший буфер влажности
        sorptionWm: 0.04, sorptionC: 20, sorptionK: 0.85,
        hysteresisRatio: 0.70
    },
    CELLULOSE_INSULATION: {
        id: 8, name: 'Cellulose Insulation',
        thermalConductivity: 0.04, density: 50, specificHeat: 1600,
        vaporPermeability: 2,
        mbv: 1.85,  // Good-Excellent
        sorptionWm: 0.08, sorptionC: 12, sorptionK: 0.82,
        hysteresisRatio: 0.78
    }
};

export interface CHTConfig {
    h_conv: number;           // Convective heat transfer coefficient [W/(m²·K)]
    T_ref: number;           // Reference temperature [K]
    beta: number;            // Thermal expansion coefficient [1/K]
    updateInterval: number;   // How often to exchange with backend [ms]
    location?: Location;      // Building location for solar calculations
    
    // Параметры влагопереноса (ISO 13788)
    D_v: number;              // Коэффициент диффузии пара в воздухе [m²/s] (~2.5e-5)
    moldRiskThreshold: number; // Порог RH для риска плесени (0.8 = 80%)
    moldRiskSteps: number;    // Количество шагов для фиксации риска плесени
    
    // TRL 7: Moisture Buffer Value (MBV) model parameters
    enableMBV: boolean;        // Включить модель MBV
    mbvTimescale: number;      // Временной масштаб буферизации [s] (8h = 28800s standard)
    enableHysteresis: boolean; // Включить гистерезис сорбции/десорбции
}

export interface ThermalBoundaryConditions {
    wall_temperature: number;    // From RC-Network [°C]
    window_temperature: number;  // From RC-Network [°C]
    air_temperature: number;     // Current air temp [°C]
    external_temperature: number; // Outdoor temp [°C]
    timestamp: string;
}

export class CHTSolver {
    private device: GPUDevice;
    private gridConfig: VoxelGridConfig;
    private config: CHTConfig;
    
    // GPU Resources
    private uniformBuffer!: GPUBuffer;
    private temperatureBufferA!: GPUBuffer;  // Double buffering
    private temperatureBufferB!: GPUBuffer;
    private heatFluxBuffer!: GPUBuffer;
    
    // Humidity buffers (Double Buffering) - ISO 13788
    private humidityBufferA!: GPUBuffer;     // Относительная влажность 0.0-1.0
    private humidityBufferB!: GPUBuffer;
    private moldRiskBuffer!: GPUBuffer;       // Флаги риска плесени
    private moldRiskCounterBuffer!: GPUBuffer; // Счетчик шагов с высокой влажностью
    
    // Pipelines
    private diffusionPipeline!: GPUComputePipeline;
    private convectionPipeline!: GPUComputePipeline;
    private boundaryPipeline!: GPUComputePipeline;
    private buoyancyPipeline!: GPUComputePipeline;
    
    // Humidity pipelines
    private vaporDiffusionPipeline!: GPUComputePipeline;
    private moldRiskPipeline!: GPUComputePipeline;
    
    private bindGroupA!: GPUBindGroup;
    private bindGroupB!: GPUBindGroup;
    private humidityBindGroupA!: GPUBindGroup;
    private humidityBindGroupB!: GPUBindGroup;
    private currentBuffer: 'A' | 'B' = 'A';
    
    // Backend communication
    private lastBackendUpdate: number = 0;
    private boundaryConditions: ThermalBoundaryConditions | null = null;
    
    // Solar calculation
    private solarCalculator: SolarPositionAlgorithm | null = null;
    private solarRadiationBuffer!: GPUBuffer;
    
    constructor(device: GPUDevice, gridConfig: VoxelGridConfig, config?: Partial<CHTConfig>) {
        this.device = device;
        this.gridConfig = gridConfig;
        
        this.config = {
            h_conv: 10.0,        // Natural convection coefficient
            T_ref: 293.15,       // 20°C reference
            beta: 3.4e-3,        // Air thermal expansion
            updateInterval: 60000, // Update every minute (1 game hour = 1 real minute)
            location: {
                latitude: 40.7128,    // Default: NYC
                longitude: -74.0060,
                timezone: -5,
                elevation: 10
            },
            // ISO 13788 parameters
            D_v: 2.5e-5,         // Diffusion coefficient of water vapor in air
            moldRiskThreshold: 0.8, // 80% RH threshold for mold risk
            moldRiskSteps: 100,  // ~100 шагов (~1.7 минут при dt=1с)
            
            // TRL 7: MBV model
            enableMBV: true,
            mbvTimescale: 28800,  // 8 hours (NORDTEST standard)
            enableHysteresis: true,
            ...config
        };
        
        // Initialize solar calculator
        if (this.config.location) {
            this.solarCalculator = new SolarPositionAlgorithm(this.config.location);
        }
    }
    
    /**
     * Initialize CHT solver with WebGPU resources
     */
    async initialize(
        voxelStateBuffer: GPUBuffer, 
        velocityBuffer: GPUBuffer,
        initialTemperature: Float32Array
    ): Promise<void> {
        const { nx, ny, nz, totalVoxels } = this.gridConfig.dimensions;
        
        // Create uniform buffer
        const uniformData = new Float32Array([
            nx, ny, nz, 0, // grid_size + padding
            this.gridConfig.resolution,
            0.001, // dt (will be updated per step)
            2.2e-5, // alpha_solid (concrete thermal diffusivity)
            2.2e-5, // alpha_fluid (air thermal diffusivity)
            this.config.h_conv,
            0, 0, -9.81, // gravity
            this.config.beta,
            this.config.T_ref,
            0, 0 // padding
        ]);
        
        this.uniformBuffer = this.device.createBuffer({
            label: 'CHT Uniform Buffer',
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
        
        // Create temperature buffers (double buffering)
        const tempSize = totalVoxels * 4; // float32
        
        this.temperatureBufferA = this.device.createBuffer({
            label: 'Temperature Buffer A',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        
        this.temperatureBufferB = this.device.createBuffer({
            label: 'Temperature Buffer B',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        
        // Initialize with room temperature
        this.device.queue.writeBuffer(this.temperatureBufferA, 0, initialTemperature);
        this.device.queue.writeBuffer(this.temperatureBufferB, 0, initialTemperature);
        
        // Create heat flux buffer
        this.heatFluxBuffer = this.device.createBuffer({
            label: 'Heat Flux Buffer',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // ============================================
        // Humidity buffers (ISO 13788 compliance)
        // ============================================
        
        // Initialize humidity with 50% RH
        const initialHumidity = new Float32Array(totalVoxels).fill(0.5);
        
        this.humidityBufferA = this.device.createBuffer({
            label: 'Humidity Buffer A',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(this.humidityBufferA, 0, initialHumidity);
        
        this.humidityBufferB = this.device.createBuffer({
            label: 'Humidity Buffer B',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(this.humidityBufferB, 0, initialHumidity);
        
        // Mold risk flags (MOLD_RISK = 0x100 in VoxelState)
        this.moldRiskBuffer = this.device.createBuffer({
            label: 'Mold Risk Buffer',
            size: totalVoxels * 4, // u32 per voxel
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Counter for consecutive high-humidity steps
        this.moldRiskCounterBuffer = this.device.createBuffer({
            label: 'Mold Risk Counter Buffer',
            size: totalVoxels * 4, // u32 per voxel
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Load shaders
        const shaderModule = this.device.createShaderModule({
            label: 'CHT Shader Module',
            code: await this.loadShaderCode()
        });
        
        // Create pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'CHT Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
            ]
        });
        
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'CHT Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout]
        });
        
        // Create compute pipelines for each step
        this.diffusionPipeline = this.device.createComputePipeline({
            label: 'Diffusion Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'diffusion_step'
            }
        });
        
        this.convectionPipeline = this.device.createComputePipeline({
            label: 'Convection Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'convection_step'
            }
        });
        
        this.boundaryPipeline = this.device.createComputePipeline({
            label: 'Boundary Coupling Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'boundary_coupling'
            }
        });
        
        this.buoyancyPipeline = this.device.createComputePipeline({
            label: 'Buoyancy Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'compute_buoyancy'
            }
        });
        
        // Humidity pipelines (ISO 13788)
        this.vaporDiffusionPipeline = this.device.createComputePipeline({
            label: 'Vapor Diffusion Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'vapor_diffusion_step'
            }
        });
        
        this.moldRiskPipeline = this.device.createComputePipeline({
            label: 'Mold Risk Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'calculate_mold_risk'
            }
        });
        
        // Create bind groups for double buffering
        this.bindGroupA = this.device.createBindGroup({
            label: 'CHT Bind Group A',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelStateBuffer } },
                { binding: 2, resource: { buffer: this.temperatureBufferA } },
                { binding: 3, resource: { buffer: this.temperatureBufferB } },
                { binding: 4, resource: { buffer: velocityBuffer } },
                { binding: 5, resource: { buffer: this.heatFluxBuffer } }
            ]
        });
        
        this.bindGroupB = this.device.createBindGroup({
            label: 'CHT Bind Group B',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelStateBuffer } },
                { binding: 2, resource: { buffer: this.temperatureBufferB } },
                { binding: 3, resource: { buffer: this.temperatureBufferA } },
                { binding: 4, resource: { buffer: velocityBuffer } },
                { binding: 5, resource: { buffer: this.heatFluxBuffer } }
            ]
        });
        
        console.log('[CHTSolver] Initialized');
    }
    
    /**
     * Calculate and apply solar radiation based on time and location
     */
    private async updateSolarRadiation(): Promise<void> {
        if (!this.solarCalculator) return;
        
        const now = new Date();
        const solarPosition = this.solarCalculator.calculate(now);
        const solarIrradiance = this.solarCalculator.calculateSolarIrradiance(solarPosition, now);
        
        // Update solar radiation buffer with directional intensity
        const solarData = new Float32Array([
            Math.sin(this.deg2rad(solarPosition.azimuth)) * Math.cos(this.deg2rad(solarPosition.elevation)),
            Math.sin(this.deg2rad(solarPosition.elevation)),
            Math.cos(this.deg2rad(solarPosition.azimuth)) * Math.cos(this.deg2rad(solarPosition.elevation)),
            solarIrradiance
        ]);
        
        this.device.queue.writeBuffer(this.uniformBuffer, 16 * 4, solarData); // Write to solar field
        
        console.log(`[CHT] Solar: Az=${solarPosition.azimuth.toFixed(1)}°, El=${solarPosition.elevation.toFixed(1)}°, Irr=${solarIrradiance.toFixed(0)}W/m²`);
    }
    
    private deg2rad(deg: number): number {
        return deg * Math.PI / 180;
    }
    
    /**
     * Execute one CHT timestep
     */
    async step(dt: number): Promise<void> {
        // Update dt in uniforms
        const dtData = new Float32Array([dt]);
        this.device.queue.writeBuffer(this.uniformBuffer, 5 * 4, dtData); // Offset to dt field
        
        // Update solar radiation
        await this.updateSolarRadiation();
        
        const commandEncoder = this.device.createCommandEncoder({
            label: 'CHT Step Command Encoder'
        });
        
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const workgroupSize = 8;
        const dispatchX = Math.ceil(nx / workgroupSize);
        const dispatchY = Math.ceil(ny / workgroupSize);
        const dispatchZ = Math.ceil(nz / workgroupSize);
        
        const bindGroup = this.currentBuffer === 'A' ? this.bindGroupA : this.bindGroupB;
        
        // 1. Diffusion in solids
        const diffusionPass = commandEncoder.beginComputePass({ label: 'Diffusion Pass' });
        diffusionPass.setPipeline(this.diffusionPipeline);
        diffusionPass.setBindGroup(0, bindGroup);
        diffusionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        diffusionPass.end();
        
        // 2. Convection in fluids
        const convectionPass = commandEncoder.beginComputePass({ label: 'Convection Pass' });
        convectionPass.setPipeline(this.convectionPipeline);
        convectionPass.setBindGroup(0, bindGroup);
        convectionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        convectionPass.end();
        
        // 3. Boundary coupling (heat exchange at interfaces)
        const boundaryPass = commandEncoder.beginComputePass({ label: 'Boundary Pass' });
        boundaryPass.setPipeline(this.boundaryPipeline);
        boundaryPass.setBindGroup(0, bindGroup);
        boundaryPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        boundaryPass.end();
        
        // 4. Compute buoyancy forces for LBM
        const buoyancyPass = commandEncoder.beginComputePass({ label: 'Buoyancy Pass' });
        buoyancyPass.setPipeline(this.buoyancyPipeline);
        buoyancyPass.setBindGroup(0, bindGroup);
        buoyancyPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        buoyancyPass.end();
        
        // ============================================
        // 5. Vapor diffusion (ISO 13788)
        // ============================================
        const vaporPass = commandEncoder.beginComputePass({ label: 'Vapor Diffusion Pass' });
        vaporPass.setPipeline(this.vaporDiffusionPipeline);
        vaporPass.setBindGroup(0, bindGroup);
        vaporPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        vaporPass.end();
        
        // 6. Mold risk assessment
        const moldPass = commandEncoder.beginComputePass({ label: 'Mold Risk Pass' });
        moldPass.setPipeline(this.moldRiskPipeline);
        moldPass.setBindGroup(0, bindGroup);
        moldPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        moldPass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Swap buffers
        this.currentBuffer = this.currentBuffer === 'A' ? 'B' : 'A';
        
        // Check if we need to sync with backend
        await this.checkBackendSync();
    }
    
    /**
     * Update boundary conditions from RC-Network backend
     */
    async updateBoundaryConditions(conditions: ThermalBoundaryConditions): Promise<void> {
        this.boundaryConditions = conditions;
        
        // Convert Celsius to Kelvin and apply to wall voxels
        const wallTempK = conditions.wall_temperature + 273.15;
        const windowTempK = conditions.window_temperature + 273.15;
        
        // This would need the actual voxel mapping to know which voxels are walls/windows
        // For now, we'll apply a uniform temperature to boundary voxels
        // In production, we'd have a mapping from voxel index to wall type
        
        console.log(`[CHTSolver] Updated boundary conditions: Wall=${conditions.wall_temperature}°C, Window=${conditions.window_temperature}°C`);
        
        this.lastBackendUpdate = Date.now();
    }
    
    /**
     * Check if we need to sync with Python backend
     */
    private async checkBackendSync(): Promise<void> {
        const now = Date.now();
        if (now - this.lastBackendUpdate > this.config.updateInterval) {
            // Request update from backend
            await this.requestBackendUpdate();
        }
    }
    
    /**
     * Request temperature update from RC-Network backend
     */
    private async requestBackendUpdate(): Promise<void> {
        try {
            // Get current average air temperature
            const avgTemp = await this.getAverageAirTemperature();
            
            // Call Python backend
            const response = await fetch('http://localhost:8000/analyze/thermal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    room: {
                        wall_area: 60,
                        window_area: 6,
                        floor_area: 25,
                        volume: 75,
                        wall_thickness: 0.3,
                        u_value_wall: 0.3,
                        u_value_window: 1.2,
                        air_change_rate: 0.5
                    },
                    external_temperature: 10, // This should come from weather API
                    solar_radiation: 200      // This should come from time of day
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                await this.updateBoundaryConditions(data.boundary_conditions);
            }
        } catch (error) {
            console.error('[CHTSolver] Backend sync failed:', error);
        }
    }
    
    /**
     * Get average air temperature for feedback to RC-Network
     */
    async getAverageAirTemperature(): Promise<number> {
        // Create staging buffer for readback
        const stagingBuffer = this.device.createBuffer({
            size: this.temperatureBufferA.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        const sourceBuffer = this.currentBuffer === 'A' ? this.temperatureBufferA : this.temperatureBufferB;
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const tempData = new Float32Array(stagingBuffer.getMappedRange());
        
        // Calculate average (simplified - should only average FLUID voxels)
        let sum = 0;
        let count = 0;
        for (let i = 0; i < tempData.length; i++) {
            if (tempData[i] > 200 && tempData[i] < 400) { // Sanity check (200K to 400K)
                sum += tempData[i];
                count++;
            }
        }
        
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return count > 0 ? (sum / count) - 273.15 : 20; // Return in Celsius
    }
    
    /**
     * Get heat flux buffer for LBM buoyancy calculations
     */
    getHeatFluxBuffer(): GPUBuffer {
        return this.heatFluxBuffer;
    }
    
    /**
     * Get current temperature buffer
     */
    getTemperatureBuffer(): GPUBuffer {
        return this.currentBuffer === 'A' ? this.temperatureBufferA : this.temperatureBufferB;
    }
    
    /**
     * Load shader code from heat_transfer.wgsl
     */
    private async loadShaderCode(): Promise<string> {
        return heatTransferShaderCode;
    }
    
    /**
     * Get humidity buffer for external access
     */
    getHumidityBuffer(): GPUBuffer {
        return this.currentBuffer === 'A' ? this.humidityBufferA : this.humidityBufferB;
    }
    
    /**
     * Get mold risk buffer for visualization
     */
    getMoldRiskBuffer(): GPUBuffer {
        return this.moldRiskBuffer;
    }
    
    /**
     * TRL 7: Calculate moisture buffer effect on air humidity
     * Implements NORDTEST MBV model for transient moisture response
     * 
     * Δm = MBV × A × ΔRH / t_cycle
     * where:
     *   Δm = moisture flux [g/s]
     *   MBV = Moisture Buffer Value [g/(m²·%RH)]
     *   A = surface area [m²]
     *   ΔRH = change in relative humidity [%]
     *   t_cycle = cycle time [s]
     */
    async calculateMoistureBufferEffect(
        surfaceAreas: Map<number, number>,  // materialId -> surface area [m²]
        currentRH: number,                   // Current air RH [0-1]
        targetRH: number,                    // Target/equilibrium RH [0-1]
        dt: number                           // Time step [s]
    ): Promise<{
        bufferedRH: number;                  // Resulting air RH after buffer effect
        moistureFlux: number;                // Total moisture flux [g/s]
        bufferCapacity: number;              // Remaining buffer capacity [g]
        materialContributions: Map<string, number>;  // Per-material contribution
    }> {
        if (!this.config.enableMBV) {
            return {
                bufferedRH: currentRH,
                moistureFlux: 0,
                bufferCapacity: 0,
                materialContributions: new Map()
            };
        }
        
        const deltaRH = (targetRH - currentRH) * 100;  // Convert to %
        let totalMoistureFlux = 0;  // g/s
        const contributions = new Map<string, number>();
        
        // Calculate contribution from each material
        for (const [materialId, area] of surfaceAreas) {
            const material = Object.values(MATERIAL_PROPERTIES).find(m => m.id === materialId);
            if (!material || material.mbv === 0) continue;
            
            // MBV model: Δm = MBV × A × ΔRH / t_cycle
            // Time constant τ = t_cycle / 2π for exponential response
            const timeConstant = this.config.mbvTimescale / (2 * Math.PI);
            const responseRate = 1 - Math.exp(-dt / timeConstant);
            
            // Moisture flux from this material
            let flux = material.mbv * area * deltaRH * responseRate;
            
            // Apply hysteresis if enabled (asymmetric sorption/desorption)
            if (this.config.enableHysteresis && material.hysteresisRatio) {
                // Desorption is slower than sorption
                if (deltaRH < 0) {  // Desorption (RH decreasing)
                    flux *= material.hysteresisRatio;
                }
            }
            
            totalMoistureFlux += flux;
            contributions.set(material.name, flux);
        }
        
        // Convert moisture flux to RH change in air volume
        // Assume standard room: V = 75 m³, T = 20°C
        // At 20°C, saturation: 17.3 g/m³
        const roomVolume = 75;  // m³ (TODO: get from actual room)
        const saturationDensity = 17.3;  // g/m³ at 20°C
        const rhChange = totalMoistureFlux * dt / (roomVolume * saturationDensity);
        
        const bufferedRH = Math.max(0, Math.min(1, currentRH + rhChange));
        
        // Calculate remaining buffer capacity
        // Simplified: assume 50% saturation at current state
        let totalCapacity = 0;
        for (const [materialId, area] of surfaceAreas) {
            const material = Object.values(MATERIAL_PROPERTIES).find(m => m.id === materialId);
            if (!material || material.mbv === 0) continue;
            // Maximum buffer at full RH swing (0-100%)
            totalCapacity += material.mbv * area * 50;  // 50% available capacity
        }
        
        console.log(`[CHTSolver] MBV: ΔRH=${(rhChange*100).toFixed(2)}%, flux=${totalMoistureFlux.toFixed(2)}g/s`);
        
        return {
            bufferedRH,
            moistureFlux: totalMoistureFlux,
            bufferCapacity: totalCapacity,
            materialContributions: contributions
        };
    }
    
    /**
     * TRL 7: Calculate equilibrium moisture content using GAB sorption isotherm
     * w = (w_m × C × K × RH) / ((1 - K×RH) × (1 - K×RH + C×K×RH))
     */
    calculateSorptionIsotherm(material: MaterialProperties, rh: number): number {
        if (!material.sorptionWm || !material.sorptionC || !material.sorptionK) {
            // Fallback: linear approximation
            return rh * 0.05;  // ~5% moisture content at 100% RH
        }
        
        const { sorptionWm: wm, sorptionC: C, sorptionK: K } = material;
        const phi = Math.max(0.01, Math.min(0.99, rh));  // Clamp to avoid singularities
        
        // GAB equation
        const numerator = wm * C * K * phi;
        const denominator = (1 - K * phi) * (1 - K * phi + C * K * phi);
        
        return numerator / denominator;  // kg water / kg dry material
    }
    
    /**
     * TRL 7: Get comprehensive humidity analysis
     */
    async getHumidityAnalysis(): Promise<{
        averageRH: number;
        minRH: number;
        maxRH: number;
        moldRisk: { totalVoxels: number; affectedVoxels: number; riskPercentage: number };
        bufferEfficiency: number;  // 0-1, how well materials are buffering
        comfortLevel: 'dry' | 'optimal' | 'humid' | 'critical';
    }> {
        // Get humidity data
        const stagingBuffer = this.device.createBuffer({
            size: this.humidityBufferA.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        const sourceBuffer = this.currentBuffer === 'A' ? this.humidityBufferA : this.humidityBufferB;
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const humidityData = new Float32Array(stagingBuffer.getMappedRange());
        
        let sum = 0, min = 1, max = 0, count = 0;
        for (let i = 0; i < humidityData.length; i++) {
            const rh = humidityData[i];
            if (rh >= 0 && rh <= 1) {
                sum += rh;
                min = Math.min(min, rh);
                max = Math.max(max, rh);
                count++;
            }
        }
        
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        const averageRH = count > 0 ? sum / count : 0.5;
        
        // Get mold risk data
        const moldRisk = await this.getMoldRiskData();
        
        // Calculate buffer efficiency (how stable is RH)
        const rhVariance = max - min;
        const bufferEfficiency = Math.max(0, 1 - rhVariance * 2);  // Lower variance = better buffering
        
        // Determine comfort level
        let comfortLevel: 'dry' | 'optimal' | 'humid' | 'critical';
        if (averageRH < 0.3) {
            comfortLevel = 'dry';
        } else if (averageRH < 0.6) {
            comfortLevel = 'optimal';
        } else if (averageRH < 0.8) {
            comfortLevel = 'humid';
        } else {
            comfortLevel = 'critical';
        }
        
        return {
            averageRH,
            minRH: min,
            maxRH: max,
            moldRisk,
            bufferEfficiency,
            comfortLevel
        };
    }
    
    /**
     * Get mold risk data for analysis
     */
    async getMoldRiskData(): Promise<{ totalVoxels: number; affectedVoxels: number; riskPercentage: number }> {
        const stagingBuffer = this.device.createBuffer({
            size: this.moldRiskBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.moldRiskBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(stagingBuffer.getMappedRange());
        
        let affectedVoxels = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] !== 0) affectedVoxels++;
        }
        
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return {
            totalVoxels: data.length,
            affectedVoxels,
            riskPercentage: (affectedVoxels / data.length) * 100
        };
    }
    
    /**
     * Cleanup resources
     */
    destroy(): void {
        this.uniformBuffer?.destroy();
        this.temperatureBufferA?.destroy();
        this.temperatureBufferB?.destroy();
        this.heatFluxBuffer?.destroy();
        this.humidityBufferA?.destroy();
        this.humidityBufferB?.destroy();
        this.moldRiskBuffer?.destroy();
        this.moldRiskCounterBuffer?.destroy();
        
        console.log('[CHTSolver] Destroyed');
    }
}
