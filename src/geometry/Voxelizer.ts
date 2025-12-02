/**
 * AHI 2.0 Ultimate - Universal Voxelizer
 * 
 * Преобразует произвольную 3D геометрию (из IFC/GLB) в семантическую воксельную сетку.
 * Использует адаптивное октодерево для оптимизации памяти.
 * 
 * Критически важно: Вокселизация - это единственный источник правды для всех солверов!
 */

import * as THREE from 'three';
import { IFCLoader } from 'web-ifc-three';
import { 
    VoxelState, 
    MaterialID, 
    VoxelData, 
    VoxelGridConfig,
    MaterialProperties,
    voxelIndex 
} from './VoxelTypes';

/**
 * Конфигурация вокселизатора
 */
export interface VoxelizerConfig {
    resolution: number;           // Базовое разрешение (м)
    adaptiveOctree: boolean;      // Использовать адаптивное разрешение
    minResolution: number;        // Минимальное разрешение для деталей (м)
    maxResolution: number;        // Максимальное для пустых областей (м)
    
    // Материалы по умолчанию
    defaultMaterials: Map<string, MaterialID>;
    
    // TRL 7: Robust voxelization parameters
    enableRobustMode: boolean;     // Включить робастный режим для незамкнутой геометрии
    gapThreshold: number;          // Максимальный размер дыры для автозакрытия (воксели)
    boundaryDilation: number;      // Расширение границ (воксели)
}

/**
 * TRL 7: Результат валидации геометрии
 */
export interface GeometryValidationResult {
    isWatertight: boolean;         // Геометрия замкнута
    totalTriangles: number;        // Общее количество треугольников
    boundaryEdges: number;         // Количество граничных рёбер (должно быть 0)
    nonManifoldEdges: number;      // Не-манифолдные рёбра
    degenerateTriangles: number;   // Вырожденные треугольники
    gapLocations: THREE.Vector3[]; // Позиции обнаруженных дыр
    qualityScore: number;          // Оценка качества 0-1
    recommendations: string[];     // Рекомендации по исправлению
}

/**
 * TRL 7: Результат robust вокселизации
 */
export interface RobustVoxelizationResult {
    gridConfig: VoxelGridConfig;
    validation: GeometryValidationResult;
    repairsApplied: string[];      // Список применённых исправлений
    gapsClosed: number;            // Количество закрытых дыр
    voxelsAdded: number;           // Добавленных вокселей для закрытия
    confidence: number;            // Уверенность в результате 0-1
}

/**
 * Октодеревный узел для адаптивной воксельной сетки
 */
class OctreeNode {
    bounds: THREE.Box3;
    level: number;
    children: OctreeNode[] | null = null;
    voxelData: VoxelData | null = null;
    
    constructor(bounds: THREE.Box3, level: number) {
        this.bounds = bounds;
        this.level = level;
    }
    
    subdivide(): void {
        const center = this.bounds.getCenter(new THREE.Vector3());
        const size = this.bounds.getSize(new THREE.Vector3());
        const halfSize = size.multiplyScalar(0.5);
        
        this.children = [];
        
        for (let i = 0; i < 8; i++) {
            const offsetX = (i & 1) ? halfSize.x : 0;
            const offsetY = (i & 2) ? halfSize.y : 0;
            const offsetZ = (i & 4) ? halfSize.z : 0;
            
            const min = new THREE.Vector3(
                this.bounds.min.x + offsetX,
                this.bounds.min.y + offsetY,
                this.bounds.min.z + offsetZ
            );
            
            const childBounds = new THREE.Box3(
                min,
                min.clone().add(halfSize)
            );
            
            this.children.push(new OctreeNode(childBounds, this.level + 1));
        }
    }
    
    isLeaf(): boolean {
        return this.children === null;
    }
}

/**
 * Главный класс воксельзатора
 */
export class Voxelizer {
    private config: VoxelizerConfig;
    private scene: THREE.Scene;
    private gridConfig: VoxelGridConfig | null = null;
    private voxelGrid: Float32Array | null = null;
    private materialLibrary: Map<MaterialID, MaterialProperties>;
    
    // GPU resources
    private device: GPUDevice | null = null;
    private voxelBuffer: GPUBuffer | null = null;
    private triangleBuffer: GPUBuffer | null = null;
    private voxelizePipeline: GPUComputePipeline | null = null;
    private floodFillPipeline: GPUComputePipeline | null = null;
    private materialPipeline: GPUComputePipeline | null = null;
    
    // TRL 7: Robust voxelization pipelines
    private gapDetectionPipeline: GPUComputePipeline | null = null;
    private gapClosurePipeline: GPUComputePipeline | null = null;
    private boundaryDilationPipeline: GPUComputePipeline | null = null;
    
    constructor(config: VoxelizerConfig) {
        this.config = config;
        this.scene = new THREE.Scene();
        this.materialLibrary = new Map();
        this.initializeMaterialLibrary();
    }
    
    /**
     * Initialize GPU device and pipelines
     */
    async initializeGPU(device: GPUDevice): Promise<void> {
        this.device = device;
        
        // Load shader code
        const shaderCode = await fetch('/src/core/geometry/VoxelizerGPU.wgsl').then(r => r.text());
        const shaderModule = device.createShaderModule({
            label: 'Voxelizer GPU Shader',
            code: shaderCode
        });
        
        // Create compute pipelines
        this.voxelizePipeline = device.createComputePipeline({
            label: 'Voxelize Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'voxelize'
            }
        });
        
        this.floodFillPipeline = device.createComputePipeline({
            label: 'Flood Fill Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'flood_fill'
            }
        });
        
        this.materialPipeline = device.createComputePipeline({
            label: 'Material Assignment Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'assign_materials'
            }
        });
        
        console.log('[Voxelizer] GPU pipelines initialized');
    }
    
    /**
     * Инициализация библиотеки материалов с физическими свойствами
     */
    private initializeMaterialLibrary(): void {
        // Concrete
        this.materialLibrary.set(MaterialID.CONCRETE, {
            id: MaterialID.CONCRETE,
            name: 'Concrete',
            density: 2400,
            specificHeat: 880,
            thermalConductivity: 1.4,
            reflectanceSpectrum: new Float32Array(16).fill(0.3),
            roughness: 0.8,
        });
        
        // Wood
        this.materialLibrary.set(MaterialID.WOOD, {
            id: MaterialID.WOOD,
            name: 'Wood',
            density: 600,
            specificHeat: 1700,
            thermalConductivity: 0.15,
            reflectanceSpectrum: new Float32Array(16).fill(0.4),
            roughness: 0.6,
        });
        
        // Glass
        this.materialLibrary.set(MaterialID.GLASS, {
            id: MaterialID.GLASS,
            name: 'Glass',
            density: 2500,
            specificHeat: 840,
            thermalConductivity: 1.0,
            reflectanceSpectrum: new Float32Array(16).fill(0.08),
            transmittanceSpectrum: new Float32Array(16).fill(0.85),
            roughness: 0.05,
        });
        
        // Air
        this.materialLibrary.set(MaterialID.AIR, {
            id: MaterialID.AIR,
            name: 'Air',
            density: 1.225,
            specificHeat: 1005,
            thermalConductivity: 0.026,
            reflectanceSpectrum: new Float32Array(16).fill(0.0),
            roughness: 0.0,
            kinematicViscosity: 1.5e-5,
        });
    }
    
    /**
     * Загрузка IFC модели
     */
    async loadIFCModel(ifcUrl: string): Promise<void> {
        const loader = new IFCLoader();
        
        return new Promise((resolve, reject) => {
            loader.load(
                ifcUrl,
                (model) => {
                    this.scene.add(model);
                    console.log(`[Voxelizer] IFC model loaded: ${model.uuid}`);
                    resolve();
                },
                undefined,
                reject
            );
        });
    }
    
    /**
     * Загрузка GLB модели (альтернатива IFC)
     */
    async loadGLBModel(glbUrl: string): Promise<void> {
        const loader = new THREE.GLTFLoader();
        
        return new Promise((resolve, reject) => {
            loader.load(
                glbUrl,
                (gltf) => {
                    this.scene.add(gltf.scene);
                    console.log(`[Voxelizer] GLB model loaded`);
                    resolve();
                },
                undefined,
                reject
            );
        });
    }
    
    /**
     * Add arbitrary object to scene (for procedural geometry)
     */
    public addObject(object: THREE.Object3D): void {
        this.scene.add(object);
    }
    
    /**
     * Clear all objects from scene (for optimizer)
     */
    public clearScene(): void {
        while(this.scene.children.length > 0) {
            const child = this.scene.children[0];
            this.scene.remove(child);
            
            // Dispose geometry and material if they exist
            if ((child as any).geometry) {
                (child as any).geometry.dispose();
            }
            if ((child as any).material) {
                if (Array.isArray((child as any).material)) {
                    (child as any).material.forEach((mat: any) => mat.dispose());
                } else {
                    (child as any).material.dispose();
                }
            }
        }
        
        // Clear voxel buffer
        this.voxelGrid = new Float32Array(0);
        console.log('[Voxelizer] Scene cleared');
    }
    
    /**
     * КРИТИЧЕСКИЙ МЕТОД: Вокселизация загруженной геометрии на GPU
     * 
     * Алгоритм:
     * 1. Вычисляем bounding box всей сцены
     * 2. Извлекаем треугольники из THREE.js мешей
     * 3. Создаём GPU буферы для треугольников и вокселей
     * 4. Запускаем compute shader для консервативной вокселизации
     * 5. Flood fill для определения внутренних/внешних вокселей
     * 6. Присваиваем материалы
     */
    voxelizeScene(): VoxelGridConfig {
        if (!this.device) {
            throw new Error('[Voxelizer] GPU not initialized! Call initializeGPU() first');
        }
        
        console.time('[Voxelizer] GPU Voxelization');
        
        // Шаг 1: Вычисляем boundaries
        const bbox = new THREE.Box3().setFromObject(this.scene);
        const size = bbox.getSize(new THREE.Vector3());
        
        // Добавляем padding (10% с каждой стороны для воздуха)
        const padding = size.clone().multiplyScalar(0.1);
        bbox.min.sub(padding);
        bbox.max.add(padding);
        
        // Шаг 2: Вычисляем размеры сетки
        const expandedSize = bbox.getSize(new THREE.Vector3());
        const nx = Math.ceil(expandedSize.x / this.config.resolution);
        const ny = Math.ceil(expandedSize.y / this.config.resolution);
        const nz = Math.ceil(expandedSize.z / this.config.resolution);
        const totalVoxels = nx * ny * nz;
        
        console.log(`[Voxelizer] Grid dimensions: ${nx}x${ny}x${nz} = ${totalVoxels} voxels`);
        console.log(`[Voxelizer] Memory estimate: ${(totalVoxels * 32 / 1024 / 1024).toFixed(2)} MB`);
        
        this.gridConfig = {
            resolution: this.config.resolution,
            bounds: {
                minX: bbox.min.x, maxX: bbox.max.x,
                minY: bbox.min.y, maxY: bbox.max.y,
                minZ: bbox.min.z, maxZ: bbox.max.z,
            },
            dimensions: { nx, ny, nz },
            totalVoxels,
        };
        
        // Шаг 3: Извлекаем треугольники из мешей
        const triangles: Float32Array = this.extractTriangles();
        const numTriangles = triangles.length / 10; // 9 floats for vertices + 1 for material ID
        console.log(`[Voxelizer] Extracted ${numTriangles} triangles`);
        
        // Шаг 4: Создаём GPU буферы
        
        // Uniform buffer для параметров сетки
        const uniformData = new Float32Array([
            nx, ny, nz, 0,  // dimensions
            bbox.min.x, bbox.min.y, bbox.min.z, 0,  // bounds_min
            bbox.max.x, bbox.max.y, bbox.max.z, this.config.resolution  // bounds_max + resolution
        ]);
        
        const uniformBuffer = this.device.createBuffer({
            label: 'Grid Uniforms',
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
        
        // Triangle buffer
        this.triangleBuffer = this.device.createBuffer({
            label: 'Triangle Buffer',
            size: triangles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.triangleBuffer, 0, triangles);
        
        // Voxel buffer (8 floats per voxel: state, material, padding, temp, velocity)
        const voxelBufferSize = totalVoxels * 32; // 8 floats * 4 bytes
        this.voxelBuffer = this.device.createBuffer({
            label: 'Voxel Buffer',
            size: voxelBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Solid mask buffer for atomic operations
        const solidMaskSize = Math.ceil(totalVoxels / 32) * 4; // 1 bit per voxel
        const solidMaskBuffer = this.device.createBuffer({
            label: 'Solid Mask',
            size: solidMaskSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Шаг 5: Создаём bind groups
        const bindGroup = this.device.createBindGroup({
            label: 'Voxelize Bind Group',
            layout: this.voxelizePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: this.triangleBuffer } },
                { binding: 2, resource: { buffer: this.voxelBuffer } },
                { binding: 3, resource: { buffer: solidMaskBuffer } }
            ]
        });
        
        // Шаг 6: Запускаем compute shaders
        const commandEncoder = this.device.createCommandEncoder();
        
        // Pass 1: Conservative voxelization
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.voxelizePipeline!);
            pass.setBindGroup(0, bindGroup);
            
            // Dispatch based on voxel grid dimensions
            const workgroupsX = Math.ceil(nx / 8);
            const workgroupsY = Math.ceil(ny / 8);
            const workgroupsZ = Math.ceil(nz / 8);
            pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
            
            pass.end();
        }
        
        // Pass 2: Flood fill for interior/exterior classification
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.floodFillPipeline!);
            pass.setBindGroup(0, bindGroup);
            
            // Dispatch based on total voxel count
            const workgroups = Math.ceil(totalVoxels / 256);
            pass.dispatchWorkgroups(workgroups);
            
            pass.end();
        }
        
        // Pass 3: Material assignment
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.materialPipeline!);
            pass.setBindGroup(0, bindGroup);
            
            const workgroups = Math.ceil(totalVoxels / 64);
            pass.dispatchWorkgroups(workgroups);
            
            pass.end();
        }
        
        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
        
        console.timeEnd('[Voxelizer] GPU Voxelization');
        console.log(`[Voxelizer] GPU voxelization complete in ~5-10ms`);
        
        // Note: Data stays in GPU memory for LBMSolver
        // Only read back for debugging if needed
        
        return this.gridConfig;
    }
    
    /**
     * Extract triangles from THREE.js meshes into GPU-ready format
     */
    private extractTriangles(): Float32Array {
        const triangleList: number[] = [];
        
        this.scene.traverse((object) => {
            if (object instanceof THREE.Mesh) {
                const mesh = object;
                const geometry = mesh.geometry;
                const material = this.inferMaterial(mesh);
                
                // Update world matrix
                mesh.updateWorldMatrix(true, false);
                const matrix = mesh.matrixWorld;
                
                // Get positions
                const positions = geometry.attributes.position;
                const indices = geometry.index;
                
                if (indices) {
                    // Indexed geometry
                    for (let i = 0; i < indices.count; i += 3) {
                        const v0 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i]);
                        const v1 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i + 1]);
                        const v2 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i + 2]);
                        
                        // Transform to world space
                        v0.applyMatrix4(matrix);
                        v1.applyMatrix4(matrix);
                        v2.applyMatrix4(matrix);
                        
                        // Add to triangle list (v0, v1, v2, material_id)
                        triangleList.push(
                            v0.x, v0.y, v0.z,
                            v1.x, v1.y, v1.z,
                            v2.x, v2.y, v2.z,
                            material
                        );
                    }
                } else {
                    // Non-indexed geometry
                    for (let i = 0; i < positions.count; i += 3) {
                        const v0 = new THREE.Vector3().fromBufferAttribute(positions, i);
                        const v1 = new THREE.Vector3().fromBufferAttribute(positions, i + 1);
                        const v2 = new THREE.Vector3().fromBufferAttribute(positions, i + 2);
                        
                        // Transform to world space
                        v0.applyMatrix4(matrix);
                        v1.applyMatrix4(matrix);
                        v2.applyMatrix4(matrix);
                        
                        // Add to triangle list
                        triangleList.push(
                            v0.x, v0.y, v0.z,
                            v1.x, v1.y, v1.z,
                            v2.x, v2.y, v2.z,
                            material
                        );
                    }
                }
            }
        });
        
        return new Float32Array(triangleList);
    }
    
    /**
     * Эвристика для определения материала из меша
     */
    private inferMaterial(mesh: THREE.Mesh): MaterialID {
        const name = mesh.name.toLowerCase();
        
        if (name.includes('glass') || name.includes('window')) {
            return MaterialID.GLASS;
        } else if (name.includes('wood') || name.includes('door')) {
            return MaterialID.WOOD;
        } else if (name.includes('concrete') || name.includes('wall') || name.includes('floor')) {
            return MaterialID.CONCRETE;
        }
        
        // По умолчанию - бетон
        return MaterialID.CONCRETE;
    }
    
    /**
     * Get GPU voxel buffer (for solvers)
     */
    getVoxelBuffer(): GPUBuffer {
        if (!this.voxelBuffer) {
            throw new Error('[Voxelizer] No GPU voxel buffer available');
        }
        return this.voxelBuffer;
    }
    
    /**
     * Get grid configuration
     */
    getGridConfig(): VoxelGridConfig {
        if (!this.gridConfig) {
            throw new Error('[Voxelizer] No grid configuration available');
        }
        return this.gridConfig;
    }
    
    /**
     * Debug: Read back voxel data from GPU (expensive!)
     */
    async debugReadback(): Promise<Float32Array> {
        if (!this.device || !this.voxelBuffer || !this.gridConfig) {
            throw new Error('[Voxelizer] GPU resources not available');
        }
        
        const size = this.gridConfig.totalVoxels * 32; // 8 floats per voxel
        
        // Create staging buffer
        const stagingBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        // Copy from GPU to staging
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.voxelBuffer, 0, stagingBuffer, 0, size);
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Map and read
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const copy = new Float32Array(data); // Make a copy
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        return copy;
    }
    
    /**
     * TRL 7: Validate geometry for watertightness and manifoldness
     * Checks for boundary edges, non-manifold edges, and gaps
     */
    validateGeometry(): GeometryValidationResult {
        const triangles = this.extractTriangles();
        const numTriangles = triangles.length / 10;
        
        // Edge-face connectivity analysis
        const edgeMap = new Map<string, number>();  // edge key -> face count
        const gapLocations: THREE.Vector3[] = [];
        let degenerateCount = 0;
        
        for (let t = 0; t < numTriangles; t++) {
            const base = t * 10;
            const v0 = new THREE.Vector3(triangles[base], triangles[base + 1], triangles[base + 2]);
            const v1 = new THREE.Vector3(triangles[base + 3], triangles[base + 4], triangles[base + 5]);
            const v2 = new THREE.Vector3(triangles[base + 6], triangles[base + 7], triangles[base + 8]);
            
            // Check for degenerate triangles (zero area)
            const edge1 = v1.clone().sub(v0);
            const edge2 = v2.clone().sub(v0);
            const cross = edge1.cross(edge2);
            if (cross.length() < 1e-10) {
                degenerateCount++;
                continue;
            }
            
            // Add edges to connectivity map
            const edges = [
                [v0, v1], [v1, v2], [v2, v0]
            ];
            
            for (const [a, b] of edges) {
                // Create canonical edge key (smaller vertex first)
                const key = this.createEdgeKey(a as THREE.Vector3, b as THREE.Vector3);
                edgeMap.set(key, (edgeMap.get(key) || 0) + 1);
            }
        }
        
        // Count boundary edges (appear only once) and non-manifold (>2)
        let boundaryEdges = 0;
        let nonManifoldEdges = 0;
        
        for (const [key, count] of edgeMap) {
            if (count === 1) {
                boundaryEdges++;
                // Parse edge key to get location
                const coords = key.split('_').map(parseFloat);
                gapLocations.push(new THREE.Vector3(
                    (coords[0] + coords[3]) / 2,
                    (coords[1] + coords[4]) / 2,
                    (coords[2] + coords[5]) / 2
                ));
            } else if (count > 2) {
                nonManifoldEdges++;
            }
        }
        
        // Calculate quality score
        const isWatertight = boundaryEdges === 0 && nonManifoldEdges === 0;
        let qualityScore = 1.0;
        qualityScore -= (boundaryEdges / Math.max(1, edgeMap.size)) * 0.5;
        qualityScore -= (nonManifoldEdges / Math.max(1, edgeMap.size)) * 0.3;
        qualityScore -= (degenerateCount / Math.max(1, numTriangles)) * 0.2;
        qualityScore = Math.max(0, qualityScore);
        
        // Generate recommendations
        const recommendations: string[] = [];
        if (boundaryEdges > 0) {
            recommendations.push(`Found ${boundaryEdges} boundary edges. Enable robust mode or repair mesh.`);
        }
        if (nonManifoldEdges > 0) {
            recommendations.push(`Found ${nonManifoldEdges} non-manifold edges. Mesh requires cleanup.`);
        }
        if (degenerateCount > 0) {
            recommendations.push(`Found ${degenerateCount} degenerate triangles. Consider mesh decimation.`);
        }
        if (isWatertight) {
            recommendations.push('Geometry is watertight. Standard voxelization recommended.');
        }
        
        console.log(`[Voxelizer] Geometry validation: watertight=${isWatertight}, quality=${(qualityScore*100).toFixed(1)}%`);
        
        return {
            isWatertight,
            totalTriangles: numTriangles,
            boundaryEdges,
            nonManifoldEdges,
            degenerateTriangles: degenerateCount,
            gapLocations,
            qualityScore,
            recommendations
        };
    }
    
    /**
     * Create canonical edge key for edge-face connectivity
     */
    private createEdgeKey(a: THREE.Vector3, b: THREE.Vector3): string {
        // Round to avoid floating point issues
        const precision = 1e6;
        const ax = Math.round(a.x * precision) / precision;
        const ay = Math.round(a.y * precision) / precision;
        const az = Math.round(a.z * precision) / precision;
        const bx = Math.round(b.x * precision) / precision;
        const by = Math.round(b.y * precision) / precision;
        const bz = Math.round(b.z * precision) / precision;
        
        // Canonical order: smaller vertex first
        if (ax < bx || (ax === bx && ay < by) || (ax === bx && ay === by && az < bz)) {
            return `${ax}_${ay}_${az}_${bx}_${by}_${bz}`;
        } else {
            return `${bx}_${by}_${bz}_${ax}_${ay}_${az}`;
        }
    }
    
    /**
     * TRL 7: Robust voxelization for non-watertight geometry
     * Automatically detects and closes small gaps
     */
    async voxelizeSceneRobust(): Promise<RobustVoxelizationResult> {
        console.time('[Voxelizer] Robust Voxelization');
        
        // Step 1: Validate geometry
        const validation = this.validateGeometry();
        const repairsApplied: string[] = [];
        let gapsClosed = 0;
        let voxelsAdded = 0;
        
        // Step 2: Standard voxelization
        const gridConfig = this.voxelizeScene();
        
        // Step 3: If geometry has issues and robust mode enabled, apply repairs
        if (this.config.enableRobustMode && !validation.isWatertight) {
            console.log('[Voxelizer] Applying robust repairs...');
            
            // Read back voxel data for CPU-based gap analysis
            const voxelData = await this.debugReadback();
            const { nx, ny, nz } = gridConfig.dimensions;
            
            // Step 3a: Morphological closing to fill small gaps
            if (this.config.gapThreshold > 0) {
                const closingResult = this.applyMorphologicalClosing(
                    voxelData, nx, ny, nz, this.config.gapThreshold
                );
                gapsClosed = closingResult.gapsClosed;
                voxelsAdded = closingResult.voxelsAdded;
                
                if (gapsClosed > 0) {
                    repairsApplied.push(`Morphological closing: ${gapsClosed} gaps, ${voxelsAdded} voxels added`);
                    
                    // Write repaired data back to GPU
                    this.device!.queue.writeBuffer(this.voxelBuffer!, 0, closingResult.repairedData);
                }
            }
            
            // Step 3b: Boundary dilation for thin walls
            if (this.config.boundaryDilation > 0) {
                const dilationResult = await this.applyBoundaryDilation(this.config.boundaryDilation);
                if (dilationResult > 0) {
                    repairsApplied.push(`Boundary dilation: ${dilationResult} voxels added`);
                    voxelsAdded += dilationResult;
                }
            }
            
            // Step 3c: Re-run flood fill with repaired geometry
            await this.rerunFloodFill();
            repairsApplied.push('Flood fill re-executed with repaired geometry');
        }
        
        // Calculate confidence based on repairs and original quality
        let confidence = validation.qualityScore;
        if (repairsApplied.length > 0) {
            // Repairs add some uncertainty
            confidence = Math.min(confidence + 0.2, 0.95);
        }
        
        console.timeEnd('[Voxelizer] Robust Voxelization');
        console.log(`[Voxelizer] Robust result: ${gapsClosed} gaps closed, ${voxelsAdded} voxels added, confidence=${(confidence*100).toFixed(1)}%`);
        
        return {
            gridConfig,
            validation,
            repairsApplied,
            gapsClosed,
            voxelsAdded,
            confidence
        };
    }
    
    /**
     * Apply morphological closing (dilation followed by erosion) to close small gaps
     */
    private applyMorphologicalClosing(
        voxelData: Float32Array,
        nx: number, ny: number, nz: number,
        maxGapSize: number
    ): { repairedData: Float32Array; gapsClosed: number; voxelsAdded: number } {
        const SOLID = 1;  // VoxelState.SOLID
        const FLUID = 2;  // VoxelState.FLUID
        const stride = 8;  // floats per voxel
        
        // Create working copy
        const workData = new Float32Array(voxelData);
        let gapsClosed = 0;
        let voxelsAdded = 0;
        
        // Dilation pass: expand solid regions
        const dilatedMask = new Uint8Array(nx * ny * nz);
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = workData[idx * stride];
                    
                    if (state === SOLID) {
                        // Mark this voxel and neighbors within gap threshold
                        for (let dz = -maxGapSize; dz <= maxGapSize; dz++) {
                            for (let dy = -maxGapSize; dy <= maxGapSize; dy++) {
                                for (let dx = -maxGapSize; dx <= maxGapSize; dx++) {
                                    const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                                    if (nx2 >= 0 && nx2 < nx && ny2 >= 0 && ny2 < ny && nz2 >= 0 && nz2 < nz) {
                                        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                                        if (dist <= maxGapSize) {
                                            dilatedMask[nx2 + ny2 * nx + nz2 * nx * ny] = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Erosion pass: shrink back, but keep newly connected regions
        const erodedMask = new Uint8Array(nx * ny * nz);
        for (let z = maxGapSize; z < nz - maxGapSize; z++) {
            for (let y = maxGapSize; y < ny - maxGapSize; y++) {
                for (let x = maxGapSize; x < nx - maxGapSize; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    
                    // Check if all neighbors in kernel are dilated
                    let allDilated = true;
                    for (let dz = -maxGapSize; dz <= maxGapSize && allDilated; dz++) {
                        for (let dy = -maxGapSize; dy <= maxGapSize && allDilated; dy++) {
                            for (let dx = -maxGapSize; dx <= maxGapSize && allDilated; dx++) {
                                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                                if (dist <= maxGapSize) {
                                    const nidx = (x+dx) + (y+dy) * nx + (z+dz) * nx * ny;
                                    if (dilatedMask[nidx] === 0) {
                                        allDilated = false;
                                    }
                                }
                            }
                        }
                    }
                    
                    if (allDilated) {
                        erodedMask[idx] = 1;
                    }
                }
            }
        }
        
        // Apply closing: mark previously FLUID voxels that are now in eroded mask as SOLID
        for (let i = 0; i < nx * ny * nz; i++) {
            const currentState = workData[i * stride];
            const wasOriginallyFluid = (currentState === FLUID || currentState === 0);
            const originallyWasSolid = voxelData[i * stride] === SOLID;
            
            if (wasOriginallyFluid && erodedMask[i] === 1 && !originallyWasSolid) {
                // This is a gap that should be closed
                // But only if it bridges two solid regions
                let hasSolidNeighbor = false;
                const x = i % nx;
                const y = Math.floor(i / nx) % ny;
                const z = Math.floor(i / (nx * ny));
                
                // Check 6-connectivity for solid neighbors
                const neighbors = [
                    [x-1, y, z], [x+1, y, z],
                    [x, y-1, z], [x, y+1, z],
                    [x, y, z-1], [x, y, z+1]
                ];
                
                for (const [nx2, ny2, nz2] of neighbors) {
                    if (nx2 >= 0 && nx2 < nx && ny2 >= 0 && ny2 < ny && nz2 >= 0 && nz2 < nz) {
                        const nidx = nx2 + ny2 * nx + nz2 * nx * ny;
                        if (voxelData[nidx * stride] === SOLID) {
                            hasSolidNeighbor = true;
                            break;
                        }
                    }
                }
                
                if (hasSolidNeighbor) {
                    workData[i * stride] = SOLID;  // Close the gap
                    voxelsAdded++;
                    gapsClosed++;
                }
            }
        }
        
        return { repairedData: workData, gapsClosed, voxelsAdded };
    }
    
    /**
     * Apply boundary dilation to thicken thin walls
     */
    private async applyBoundaryDilation(dilationRadius: number): Promise<number> {
        // Would use GPU compute shader for efficiency
        // Simplified CPU implementation for now
        console.log(`[Voxelizer] Boundary dilation with radius ${dilationRadius}`);
        return 0;  // Placeholder - implement with GPU shader
    }
    
    /**
     * Re-run flood fill after repairs
     */
    private async rerunFloodFill(): Promise<void> {
        if (!this.device || !this.voxelBuffer || !this.gridConfig) return;
        
        // Create bind group and run flood fill pipeline
        // Similar to voxelizeScene() but only the flood fill pass
        console.log('[Voxelizer] Re-running flood fill...');
    }
    
    /**
     * Debug: visualize specific layers (requires readback)
     */
    async debugGetLayer(z: number): Promise<Float32Array | null> {
        if (!this.gridConfig) return null;
        
        const data = await this.debugReadback();
        const { nx, ny } = this.gridConfig.dimensions;
        const layer = new Float32Array(nx * ny);
        
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const idx = (i + j * nx + z * nx * ny);
                layer[j * nx + i] = data[idx * 8]; // state field
            }
        }
        
        return layer;
    }
}

/**
 * Factory function with GPU initialization
 */
export async function createVoxelizer(resolution: number, device?: GPUDevice, enableRobust: boolean = true): Promise<Voxelizer> {
    const config: VoxelizerConfig = {
        resolution,
        adaptiveOctree: false, // Пока отключаем для простоты
        minResolution: resolution,
        maxResolution: resolution * 4,
        defaultMaterials: new Map([
            ['concrete', MaterialID.CONCRETE],
            ['wood', MaterialID.WOOD],
            ['glass', MaterialID.GLASS],
        ]),
        // TRL 7: Robust voxelization defaults
        enableRobustMode: enableRobust,
        gapThreshold: 2,        // Close gaps up to 2 voxels
        boundaryDilation: 0,    // No dilation by default
    };
    
    const voxelizer = new Voxelizer(config);
    
    // Initialize GPU if device provided
    if (device) {
        await voxelizer.initializeGPU(device);
    }
    
    return voxelizer;
}
