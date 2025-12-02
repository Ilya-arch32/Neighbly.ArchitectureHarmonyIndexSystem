/**
 * AHI 2.0 Ultimate - NSGA-III Multi-Objective Optimization Engine
 * 
 * The "Brain" that evolves architectural designs toward Pareto-optimal solutions
 * Balances physics (thermal, airflow), optics (lighting), and aesthetics (fractals, space syntax)
 */

import { SimulationManager } from './SimulationManager';
import { VoxelGridConfig } from './VoxelTypes';
import { SpectralTracer, LightingMetrics } from './SpectralTracer';
import { VoxelFractalAnalyzer, FractalMetrics3D } from './VoxelFractalAnalyzer';
import { SpaceSyntaxSolver, SpaceSyntaxMetrics } from './SpaceSyntaxSolver';

export interface DesignGenome {
    id: string;
    genes: {
        // Geometry genes
        wallPositions: Float32Array;      // Wall placement coordinates
        windowPositions: Float32Array;    // Window locations and sizes
        windowGlazingType: number[];      // Material properties
        wallMaterials: number[];          // Thermal properties
        
        // HVAC genes (future)
        ventPositions?: Float32Array;     // Ventilation locations
        heatingPower?: number;            // Heating capacity
        
        // Spatial genes
        partitionWalls: Float32Array;     // Interior divisions
        doorPositions: Float32Array;      // Connectivity
    };
    
    // Cached fitness values
    objectives?: ObjectiveValues;
    dominationRank?: number;
    crowdingDistance?: number;
}

export interface ObjectiveValues {
    // Physics objectives
    thermalComfort: number;        // PMV index (-0.5 to +0.5 ideal)
    energyEfficiency: number;       // kWh/m²/year
    airQuality: number;            // CO2 ppm, ventilation rate
    
    // Lighting objectives  
    daylightAutonomy: number;      // % hours > 300 lux
    circadianStimulus: number;     // Average CS value
    glareIndex: number;            // DGP < 0.35
    
    // Aesthetic objectives
    fractalDimension: number;      // Distance from 1.4 ideal
    spatialIntegration: number;    // Space Syntax integration
    viewQuality: number;           // Isovist properties
    
    // Cost objective
    constructionCost: number;      // Material + labor estimate
}

export interface NSGA3Config {
    populationSize: number;
    maxGenerations: number;
    crossoverRate: number;
    mutationRate: number;
    referencePoints: number;       // Number of reference directions
    objectives: string[];          // Which objectives to optimize
}

export class NSGA3Optimizer {
    private config: NSGA3Config;
    private simulationManager: SimulationManager;
    private device: GPUDevice;
    
    // Optimization state
    private population: DesignGenome[] = [];
    private generation: number = 0;
    private paretoFront: DesignGenome[] = [];
    private referenceDirections: Float32Array;
    
    // Analysis modules
    private spectralTracer!: SpectralTracer;
    private fractalAnalyzer!: VoxelFractalAnalyzer;
    private spaceSyntaxSolver!: SpaceSyntaxSolver;
    
    constructor(
        device: GPUDevice,
        simulationManager: SimulationManager,
        config?: Partial<NSGA3Config>
    ) {
        this.device = device;
        this.simulationManager = simulationManager;
        
        this.config = {
            populationSize: 100,
            maxGenerations: 50,
            crossoverRate: 0.9,
            mutationRate: 0.1,
            referencePoints: 91,  // H=12 for 10 objectives
            objectives: [
                'thermalComfort', 'energyEfficiency', 'daylightAutonomy',
                'circadianStimulus', 'fractalDimension', 'spatialIntegration'
            ],
            ...config
        };
        
        // Generate Das-Dennis reference points
        this.referenceDirections = this.generateReferenceDirections();
    }
    
    /**
     * Initialize optimization modules
     */
    async initialize(gridConfig: VoxelGridConfig): Promise<void> {
        // Initialize analysis modules
        this.spectralTracer = new SpectralTracer(this.device, gridConfig);
        this.fractalAnalyzer = new VoxelFractalAnalyzer(this.device, gridConfig);
        this.spaceSyntaxSolver = new SpaceSyntaxSolver(this.device, gridConfig);
        
        await Promise.all([
            this.spectralTracer.initialize(
                this.simulationManager.getVoxelBuffer(),
                [] // Material library
            ),
            this.fractalAnalyzer.initialize(),
            this.spaceSyntaxSolver.initialize()
        ]);
        
        // Generate initial population
        this.population = this.generateInitialPopulation();
        
        console.log('[NSGA3] Initialized with', this.config.populationSize, 'individuals');
    }
    
    /**
     * Run optimization for one generation
     */
    async evolveGeneration(): Promise<void> {
        console.log(`[NSGA3] Generation ${this.generation + 1}/${this.config.maxGenerations}`);
        
        // 1. Evaluate fitness for new individuals
        await this.evaluatePopulation();
        
        // 2. Non-dominated sorting
        this.nonDominatedSort();
        
        // 3. Reference point association
        this.associateReferencePoints();
        
        // 4. Environmental selection
        const parents = this.environmentalSelection();
        
        // 5. Generate offspring via crossover and mutation
        const offspring = this.generateOffspring(parents);
        
        // 6. Combine populations
        this.population = [...parents, ...offspring];
        
        // 7. Update Pareto front
        this.updateParetoFront();
        
        this.generation++;
        
        // Log best solutions
        this.logBestSolutions();
    }
    
    /**
     * Run full optimization
     */
    async optimize(): Promise<DesignGenome[]> {
        console.log('[NSGA3] Starting optimization...');
        
        for (let gen = 0; gen < this.config.maxGenerations; gen++) {
            await this.evolveGeneration();
            
            // Early stopping if convergence detected
            if (this.hasConverged()) {
                console.log('[NSGA3] Converged at generation', gen);
                break;
            }
        }
        
        console.log('[NSGA3] Optimization complete. Pareto front size:', this.paretoFront.length);
        return this.paretoFront;
    }
    
    /**
     * Generate initial random population
     */
    private generateInitialPopulation(): DesignGenome[] {
        const population: DesignGenome[] = [];
        
        for (let i = 0; i < this.config.populationSize; i++) {
            population.push(this.createRandomGenome(`ind_${i}`));
        }
        
        return population;
    }
    
    /**
     * Create random design genome
     */
    private createRandomGenome(id: string): DesignGenome {
        // Random wall positions (4 walls + partitions)
        const numWalls = 4 + Math.floor(Math.random() * 3);
        const wallPositions = new Float32Array(numWalls * 6); // x,y,z,width,height,thickness
        
        for (let i = 0; i < numWalls; i++) {
            const offset = i * 6;
            wallPositions[offset] = Math.random() * 10;     // x
            wallPositions[offset + 1] = Math.random() * 10; // y
            wallPositions[offset + 2] = 0;                  // z
            wallPositions[offset + 3] = 2 + Math.random() * 5; // width
            wallPositions[offset + 4] = 3;                  // height
            wallPositions[offset + 5] = 0.2;                // thickness
        }
        
        // Random windows
        const numWindows = 2 + Math.floor(Math.random() * 4);
        const windowPositions = new Float32Array(numWindows * 4); // x,y,width,height
        
        for (let i = 0; i < numWindows; i++) {
            const offset = i * 4;
            windowPositions[offset] = Math.random() * 10;     // x
            windowPositions[offset + 1] = 0.8 + Math.random() * 1.5; // y (above floor)
            windowPositions[offset + 2] = 1 + Math.random() * 2;    // width
            windowPositions[offset + 3] = 1 + Math.random() * 1.5;  // height
        }
        
        // Random materials
        const windowGlazingType = Array(numWindows).fill(0).map(() => Math.floor(Math.random() * 3));
        const wallMaterials = Array(numWalls).fill(0).map(() => Math.floor(Math.random() * 5));
        
        // Doors for connectivity
        const doorPositions = new Float32Array(8); // 2 doors
        doorPositions[0] = 2; doorPositions[1] = 0; doorPositions[2] = 1; doorPositions[3] = 2.1;
        doorPositions[4] = 5; doorPositions[5] = 0; doorPositions[6] = 1; doorPositions[7] = 2.1;
        
        return {
            id,
            genes: {
                wallPositions,
                windowPositions,
                windowGlazingType,
                wallMaterials,
                partitionWalls: new Float32Array(0),
                doorPositions
            }
        };
    }
    
    /**
     * Evaluate fitness objectives for population
     */
    private async evaluatePopulation(): Promise<void> {
        for (const individual of this.population) {
            if (!individual.objectives) {
                individual.objectives = await this.evaluateDesign(individual);
            }
        }
    }
    
    /**
     * Evaluate single design
     */
    private async evaluateDesign(genome: DesignGenome): Promise<ObjectiveValues> {
        // Apply genome to simulation
        await this.applyGenomeToSimulation(genome);
        
        // Run simulation for thermal equilibrium
        for (let i = 0; i < 100; i++) {
            await this.simulationManager.step();
        }
        
        // Get simulation snapshot
        const snapshot = await this.simulationManager.getSnapshot();
        
        // Evaluate thermal comfort (PMV model)
        const thermalComfort = this.evaluateThermalComfort(snapshot);
        
        // Evaluate energy efficiency
        const energyEfficiency = this.evaluateEnergyEfficiency(snapshot);
        
        // Evaluate air quality
        const airQuality = this.evaluateAirQuality(snapshot);
        
        // Evaluate lighting
        const lightingMetrics = await this.evaluateLighting();
        
        // Evaluate aesthetics
        const aestheticMetrics = await this.evaluateAesthetics();
        
        // Estimate construction cost
        const constructionCost = this.estimateCost(genome);
        
        return {
            thermalComfort,
            energyEfficiency,
            airQuality,
            daylightAutonomy: lightingMetrics.daylightAutonomy,
            circadianStimulus: lightingMetrics.circadianStimulus,
            glareIndex: lightingMetrics.glareIndex,
            fractalDimension: aestheticMetrics.fractalDimension,
            spatialIntegration: aestheticMetrics.spatialIntegration,
            viewQuality: aestheticMetrics.viewQuality,
            constructionCost
        };
    }
    
    private async applyGenomeToSimulation(genome: DesignGenome): Promise<void> {
        console.log(`[NSGA3] Applying genome ${genome.id} to simulation`);
    }
    
    private evaluateThermalComfort(snapshot: any): number {
        const avgTemp = snapshot.metrics.avgTemperature;
        const avgVel = snapshot.metrics.maxVelocity;
        const pmv = (avgTemp - 22) * 0.1 - avgVel * 0.5;
        return Math.max(0, 1 - Math.abs(pmv) / 3);
    }
    
    private evaluateEnergyEfficiency(snapshot: any): number {
        const heatLoss = Math.abs(snapshot.metrics.avgTemperature - 20) * 10;
        const energyUse = heatLoss * 8760 / 1000;
        return Math.max(0, 1 - energyUse / 200);
    }
    
    private evaluateAirQuality(snapshot: any): number {
        const flowComplexity = snapshot.metrics.flowComplexityIndex || 0.5;
        return 1 - Math.abs(flowComplexity - 0.5) * 2;
    }
    
    private async evaluateLighting(): Promise<any> {
        const samplePoint = { x: 5, y: 5, z: 1.2 };
        const metrics = await this.spectralTracer.computeMetrics(samplePoint);
        
        return {
            daylightAutonomy: Math.min(1, metrics.illuminance / 300),
            circadianStimulus: metrics.circadianStimulus / 0.3,
            glareIndex: Math.max(0, 1 - metrics.illuminance / 3000)
        };
    }
    
    private async evaluateAesthetics(): Promise<any> {
        const voxelBuffer = this.simulationManager.getVoxelBuffer();
        const fractalMetrics = await this.fractalAnalyzer.analyze(voxelBuffer);
        const spaceMetrics = await this.spaceSyntaxSolver.analyze(voxelBuffer);
        
        return {
            fractalDimension: 1 - Math.abs(fractalMetrics.fractalDimension - 1.4) / 0.5,
            spatialIntegration: spaceMetrics.intelligibility,
            viewQuality: Math.min(1, spaceMetrics.visualComplexity / 10)
        };
    }
    
    private estimateCost(genome: DesignGenome): number {
        const wallArea = genome.genes.wallPositions.length / 6 * 15;
        const windowArea = genome.genes.windowPositions.length / 4 * 2;
        const totalCost = wallArea * 100 + windowArea * 300;
        return Math.max(0, 1 - totalCost / 10000);
    }
    
    private nonDominatedSort(): void {
        const n = this.population.length;
        
        for (let i = 0; i < n; i++) {
            this.population[i].dominationRank = 0;
        }
        
        const fronts: DesignGenome[][] = [[]];
        
        for (let i = 0; i < n; i++) {
            const p = this.population[i];
            let dominated = false;
            
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                
                if (this.dominates(this.population[j], p)) {
                    dominated = true;
                    break;
                }
            }
            
            if (!dominated) {
                p.dominationRank = 0;
                fronts[0].push(p);
            }
        }
    }
    
    private dominates(a: DesignGenome, b: DesignGenome): boolean {
        if (!a.objectives || !b.objectives) return false;
        
        let better = false;
        let worse = false;
        
        for (const obj of this.config.objectives) {
            const aVal = (a.objectives as any)[obj];
            const bVal = (b.objectives as any)[obj];
            
            if (aVal > bVal) better = true;
            if (aVal < bVal) worse = true;
        }
        
        return better && !worse;
    }
    
    private associateReferencePoints(): void {
        for (const individual of this.population) {
            let minDist = Infinity;
            
            for (let r = 0; r < this.referenceDirections.length / this.config.objectives.length; r++) {
                const dist = this.calculateDistance(individual, r);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            
            individual.crowdingDistance = minDist;
        }
    }
    
    private calculateDistance(individual: DesignGenome, refIndex: number): number {
        let sum = 0;
        let idx = 0;
        
        for (const obj of this.config.objectives) {
            const value = (individual.objectives as any)[obj];
            const refValue = this.referenceDirections[refIndex * this.config.objectives.length + idx];
            sum += Math.pow(value - refValue, 2);
            idx++;
        }
        
        return Math.sqrt(sum);
    }
    
    private environmentalSelection(): DesignGenome[] {
        this.population.sort((a, b) => {
            if (a.dominationRank !== b.dominationRank) {
                return (a.dominationRank || 0) - (b.dominationRank || 0);
            }
            return (b.crowdingDistance || 0) - (a.crowdingDistance || 0);
        });
        
        return this.population.slice(0, Math.floor(this.config.populationSize / 2));
    }
    
    private generateOffspring(parents: DesignGenome[]): DesignGenome[] {
        const offspring: DesignGenome[] = [];
        
        while (offspring.length < parents.length) {
            const p1 = this.tournamentSelect(parents);
            const p2 = this.tournamentSelect(parents);
            
            if (Math.random() < this.config.crossoverRate) {
                const [c1, c2] = this.crossover(p1, p2);
                offspring.push(c1, c2);
            } else {
                offspring.push(this.clone(p1), this.clone(p2));
            }
        }
        
        for (const child of offspring) {
            if (Math.random() < this.config.mutationRate) {
                this.mutate(child);
            }
        }
        
        return offspring;
    }
    
    private tournamentSelect(population: DesignGenome[]): DesignGenome {
        const tournamentSize = 3;
        let best = population[Math.floor(Math.random() * population.length)];
        
        for (let i = 1; i < tournamentSize; i++) {
            const candidate = population[Math.floor(Math.random() * population.length)];
            if ((candidate.dominationRank || 0) < (best.dominationRank || 0)) {
                best = candidate;
            }
        }
        
        return best;
    }
    
    private crossover(p1: DesignGenome, p2: DesignGenome): [DesignGenome, DesignGenome] {
        const c1 = this.clone(p1);
        const c2 = this.clone(p2);
        
        // Simulated Binary Crossover (SBX)
        const eta = 20;
        
        for (let i = 0; i < c1.genes.wallPositions.length; i++) {
            if (Math.random() < 0.5) {
                const y1 = c1.genes.wallPositions[i];
                const y2 = c2.genes.wallPositions[i];
                
                const u = Math.random();
                const beta = u <= 0.5 
                    ? Math.pow(2 * u, 1 / (eta + 1))
                    : Math.pow(1 / (2 * (1 - u)), 1 / (eta + 1));
                
                c1.genes.wallPositions[i] = 0.5 * ((1 + beta) * y1 + (1 - beta) * y2);
                c2.genes.wallPositions[i] = 0.5 * ((1 - beta) * y1 + (1 + beta) * y2);
            }
        }
        
        c1.objectives = undefined;
        c2.objectives = undefined;
        c1.id = `${p1.id}_${p2.id}_c1`;
        c2.id = `${p1.id}_${p2.id}_c2`;
        
        return [c1, c2];
    }
    
    private mutate(genome: DesignGenome): void {
        const eta = 20;
        
        for (let i = 0; i < genome.genes.wallPositions.length; i++) {
            if (Math.random() < 0.1) {
                const y = genome.genes.wallPositions[i];
                const delta = Math.random() < 0.5
                    ? Math.pow(2 * Math.random(), 1 / (eta + 1)) - 1
                    : 1 - Math.pow(2 * (1 - Math.random()), 1 / (eta + 1));
                
                genome.genes.wallPositions[i] = Math.max(0, Math.min(10, y + delta * 2));
            }
        }
        
        genome.objectives = undefined;
        genome.id += '_mut';
    }
    
    private clone(genome: DesignGenome): DesignGenome {
        return {
            id: genome.id + '_clone',
            genes: {
                wallPositions: new Float32Array(genome.genes.wallPositions),
                windowPositions: new Float32Array(genome.genes.windowPositions),
                windowGlazingType: [...genome.genes.windowGlazingType],
                wallMaterials: [...genome.genes.wallMaterials],
                partitionWalls: new Float32Array(genome.genes.partitionWalls),
                doorPositions: new Float32Array(genome.genes.doorPositions)
            },
            objectives: genome.objectives ? { ...genome.objectives } : undefined,
            dominationRank: genome.dominationRank,
            crowdingDistance: genome.crowdingDistance
        };
    }
    
    private updateParetoFront(): void {
        this.paretoFront = this.population.filter(ind => ind.dominationRank === 0);
    }
    
    private hasConverged(): boolean {
        return this.generation > 10 && this.paretoFront.length > 0 && this.generation > 30;
    }
    
    private logBestSolutions(): void {
        const best = this.paretoFront.slice(0, 3);
        
        console.log(`[NSGA3] Generation ${this.generation} - Top solutions:`);
        for (const solution of best) {
            if (solution.objectives) {
                console.log(`  ${solution.id}:`, 
                    `Thermal=${solution.objectives.thermalComfort.toFixed(2)}`,
                    `Energy=${solution.objectives.energyEfficiency.toFixed(2)}`,
                    `Daylight=${solution.objectives.daylightAutonomy.toFixed(2)}`,
                    `Fractal=${solution.objectives.fractalDimension.toFixed(2)}`
                );
            }
        }
    }
    
    private generateReferenceDirections(): Float32Array {
        const M = this.config.objectives.length;
        const p = 12; // Das-Dennis divisions parameter
        
        // Generate reference points on unit simplex
        const points = this.generateDasDennisPoints(M, p);
        
        return new Float32Array(points);
    }
    
    /**
     * Generate Das-Dennis structured reference points
     */
    private generateDasDennisPoints(M: number, p: number): number[] {
        const points: number[] = [];
        const combinations = this.generateCombinations(M, p);
        
        for (const combo of combinations) {
            // Normalize to unit simplex
            const point = combo.map(v => v / p);
            points.push(...point);
        }
        
        return points;
    }
    
    /**
     * Generate integer combinations that sum to total
     */
    private generateCombinations(dims: number, total: number): number[][] {
        if (dims === 1) {
            return [[total]];
        }
        
        const results: number[][] = [];
        
        for (let i = 0; i <= total; i++) {
            const subCombos = this.generateCombinations(dims - 1, total - i);
            for (const sub of subCombos) {
                results.push([i, ...sub]);
            }
        }
        
        return results;
    }
    
    /**
     * Tiered fitness evaluation for computational efficiency
     */
    async evaluatePopulationTiered(): Promise<void> {
        // Level 1: Quick proxy evaluation (100 individuals)
        console.log('[NSGA3] Level 1: Proxy evaluation...');
        for (const individual of this.population) {
            if (!individual.objectives) {
                individual.objectives = await this.evaluateProxy(individual);
            }
        }
        
        // Select top 20% for detailed evaluation
        this.nonDominatedSort();
        const elite = this.population
            .sort((a, b) => (a.dominationRank || 0) - (b.dominationRank || 0))
            .slice(0, Math.floor(this.population.length * 0.2));
        
        // Level 2: Full physics simulation for elite
        console.log('[NSGA3] Level 2: Full evaluation for elite...');
        for (const individual of elite) {
            individual.objectives = await this.evaluateDesign(individual);
        }
    }
    
    /**
     * Fast proxy evaluation using simplified models
     */
    private async evaluateProxy(genome: DesignGenome): Promise<ObjectiveValues> {
        // Use RC-Network instead of full CFD
        const thermalProxy = this.evaluateThermalProxy(genome);
        
        // Use view factor instead of path tracing
        const daylightProxy = this.evaluateDaylightProxy(genome);
        
        // Use 2D fractal instead of 3D
        const aestheticProxy = this.evaluateAestheticProxy(genome);
        
        // Simple cost model
        const costProxy = this.estimateCost(genome);
        
        return {
            thermalComfort: thermalProxy,
            energyEfficiency: thermalProxy * 0.9,
            airQuality: 0.7,
            daylightAutonomy: daylightProxy,
            circadianStimulus: daylightProxy * 0.8,
            glareIndex: 1 - daylightProxy * 0.3,
            fractalDimension: aestheticProxy,
            spatialIntegration: 0.7,
            viewQuality: 0.6,
            constructionCost: costProxy
        };
    }
    
    private evaluateThermalProxy(genome: DesignGenome): number {
        // Simplified U-value calculation
        const wallArea = genome.genes.wallPositions.length / 6 * 15;
        const windowArea = genome.genes.windowPositions.length / 4 * 2;
        const uValue = (wallArea * 0.3 + windowArea * 2.0) / (wallArea + windowArea);
        return Math.max(0, 1 - uValue / 3);
    }
    
    private evaluateDaylightProxy(genome: DesignGenome): number {
        // Window-to-wall ratio
        const wallArea = genome.genes.wallPositions.length / 6 * 15;
        const windowArea = genome.genes.windowPositions.length / 4 * 2;
        const wwr = windowArea / (wallArea + windowArea);
        return Math.min(1, wwr * 3);
    }
    
    private evaluateAestheticProxy(genome: DesignGenome): number {
        // Complexity from geometry variation
        const numElements = genome.genes.wallPositions.length / 6 + 
                          genome.genes.windowPositions.length / 4;
        const complexity = Math.log2(numElements + 1) / 3;
        return 1 - Math.abs(complexity - 0.7); // Optimal at moderate complexity
    }
    
    /**
     * Get best solution for specific objective
     */
    getBestForObjective(objective: string): DesignGenome | null {
        let best: DesignGenome | null = null;
        let bestValue = -Infinity;
        
        for (const ind of this.paretoFront) {
            if (ind.objectives) {
                const value = (ind.objectives as any)[objective];
                if (value > bestValue) {
                    bestValue = value;
                    best = ind;
                }
            }
        }
        
        return best;
    }
    
    /**
     * Export Pareto front for visualization
     */
    exportParetoFront(): any[] {
        return this.paretoFront.map(ind => ({
            id: ind.id,
            objectives: ind.objectives,
            genes: {
                numWalls: ind.genes.wallPositions.length / 6,
                numWindows: ind.genes.windowPositions.length / 4
            }
        }));
    }
}
