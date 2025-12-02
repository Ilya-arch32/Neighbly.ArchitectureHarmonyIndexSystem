/**
 * AHI 2.0 Ultimate - Core Voxel Data Structures
 * Lead Architect: Systems Design Team
 * 
 * Унифицированные типы для воксельной топологии - основа всей физики
 */

/**
 * Битовые маски состояния вокселя (для эффективного хранения в GPU buffers)
 */
export enum VoxelState {
    EMPTY = 0x00,           // Пустой воксель (вне геометрии)
    SOLID = 0x01,           // Твердое тело (стена, пол, мебель)
    FLUID = 0x02,           // Воздух (активная зона для LBM)
    GLASS = 0x04,           // Прозрачный материал (для оптики)
    BOUNDARY = 0x08,        // Граничный воксель (для CHT coupling)
}

/**
 * Идентификаторы материалов (расширяемый enum)
 */
export enum MaterialID {
    AIR = 0,
    CONCRETE = 1,
    WOOD = 2,
    GLASS = 3,
    INSULATION = 4,
    STEEL = 5,
    BRICK = 6,
    GYPSUM = 7,
}

/**
 * Физические свойства материала (для CHT и оптики)
 */
export interface MaterialProperties {
    id: MaterialID;
    name: string;
    
    // Thermal properties
    density: number;              // кг/м³
    specificHeat: number;         // Дж/(кг·К)
    thermalConductivity: number;  // Вт/(м·К)
    
    // Optical properties (spectral)
    reflectanceSpectrum: Float32Array;  // 16 bins, 380-780nm
    transmittanceSpectrum?: Float32Array; // Для стекла
    roughness: number;            // 0-1, для BRDF
    
    // Fluid properties (если это воздух)
    kinematicViscosity?: number;  // м²/с
}

/**
 * Структура данных для одного вокселя в памяти GPU
 * Оптимизирована для 128-битного выравнивания (5x vec4 = 80 bytes)
 */
export interface VoxelData {
    // 16 bytes - State & Material
    state: number;               // uint8 - VoxelState flags
    materialId: number;          // uint8 - MaterialID
    padding1: number;            // uint8 - Reserved
    padding2: number;            // uint8 - Reserved
    
    // 16 bytes - Thermal state
    temperature: number;         // float32 - Текущая температура (К)
    thermalMass: number;         // float32 - C_m (Дж/К)
    heatFlux: number;           // float32 - Тепловой поток (Вт)
    humidity: number;           // float32 - Относительная влажность (0-1) [ISO 13788]
    
    // 16 bytes - Fluid velocity (LBM)
    velocityX: number;          // float32 - Компонента скорости X (м/с)
    velocityY: number;          // float32 - Компонента скорости Y (м/с)
    velocityZ: number;          // float32 - Компонента скорости Z (м/с)
    pressure: number;           // float32 - Давление (Па)
    
    // 16 bytes - Optical & Environmental
    illuminance: number;        // float32 - Освещенность (лк)
    co2Concentration: number;   // float32 - Концентрация CO₂ (ppm)
    pmv: number;               // float32 - PMV индекс комфорта
    dgp: number;               // float32 - Daylight Glare Probability
    
    // 16 bytes - Neuroaesthetic Layer (Word 5)
    // Данные для расчета "карты гармонии" - визуализация нейроэстетических метрик
    visualEntropy: number;      // float32 - Локальная визуальная сложность (бит)
    fractalDimension: number;   // float32 - Локальный вклад в фрактальность D (1.0-2.0)
    isovistVolume: number;      // float32 - Объем видимого пространства (Prospect)
    integrationValue: number;   // float32 - Глобальная связность (Space Syntax)
}

/**
 * Параметры воксельной сетки
 */
export interface VoxelGridConfig {
    resolution: number;         // Размер вокселя (м), например 0.05 = 5см
    bounds: {
        minX: number; maxX: number;
        minY: number; maxY: number;
        minZ: number; maxZ: number;
    };
    dimensions: {
        nx: number; ny: number; nz: number;
    };
    totalVoxels: number;
}

/**
 * Результат воксельной симуляции (для визуализации)
 */
export interface SimulationSnapshot {
    timestamp: number;          // Время симуляции (с)
    gridConfig: VoxelGridConfig;
    
    // Поля для визуализации (Float32Array для эффективности)
    temperatureField: Float32Array;
    velocityField: Float32Array;  // [vx, vy, vz] * totalVoxels
    comfortField: Float32Array;   // PMV/PPD
    humidityField: Float32Array;  // Относительная влажность (ISO 13788)
    
    // Нейроэстетические поля (для "карты гармонии")
    neuroaestheticFields: {
        fractalDimensionField: Float32Array;   // D для каждого вокселя
        visualEntropyField: Float32Array;      // Энтропия для каждого вокселя
        isovistField: Float32Array;            // Объем изовистов
        integrationField: Float32Array;        // Space Syntax integration
    };
    
    // Агрегированные метрики
    metrics: {
        avgTemperature: number;
        maxVelocity: number;
        avgPMV: number;
        co2Max: number;
        energyConsumption: number;  // kWh
        flowComplexityIndex?: number; // 0.0 - 1.0 (Chaos Theory)
    };
    
    // Нейроэстетические метрики (для AHI формулы)
    neuroaestheticMetrics: NeuroaestheticMetrics;
}

/**
 * Метрики нейроэстетики для расчета Индекса Архитектурной Гармонии
 */
export interface NeuroaestheticMetrics {
    // Биофильная флюентность (D ≈ 1.3-1.5 = оптимум)
    fractalDimension: number;        // Глобальный D методом Box-Counting
    fractalOptimality: number;       // 1 - |D - 1.4| (близость к природному фракталу)
    
    // Когнитивная нагрузка
    visualEntropy: number;           // Глобальная визуальная энтропия (бит)
    entropyOptimality: number;       // Оптимум: 4-6 бит (не скучно, не хаос)
    
    // Пространственная понятность (Space Syntax)
    avgIsovistVolume: number;        // Средний объем изовистов (Prospect)
    avgOcclusivity: number;          // Степень закрытости (Refuge)
    prospectRefugeBalance: number;   // Баланс Prospect/Refuge
    avgIntegration: number;          // Средняя интеграция (понятность)
    intelligibility: number;         // Корреляция connectivity/integration
    
    // Примечание: Финальные оценки (AHI) рассчитываются на бэкенде
}

/**
 * Константы физических моделей
 */
export const PHYSICS_CONSTANTS = {
    // LBM D3Q19
    LBM_RELAXATION_TIME: 0.6,       // τ для BGK оператора
    LBM_LATTICE_SPEED: 0.1,         // Скорость решетки (м/с)
    
    // CHT
    STEFAN_BOLTZMANN: 5.670374419e-8, // Вт/(м²·К⁴)
    AIR_DENSITY: 1.225,              // кг/м³ при 15°C
    AIR_SPECIFIC_HEAT: 1005,         // Дж/(кг·К)
    
    // Comfort
    PMV_METABOLIC_RATE: 1.2,        // met (сидячая работа)
    PPD_THRESHOLD: 10,              // % недовольных (целевой комфорт)
    
    // Spectral optics
    SPECTRAL_BINS: 16,              // Дискретизация спектра
    WAVELENGTH_MIN: 380,            // нм
    WAVELENGTH_MAX: 780,            // нм
    
    // Neuroaesthetics (Research-based optimal values)
    FRACTAL_OPTIMAL_D: 1.4,         // Оптимальная фрактальная размерность (Taylor et al.)
    FRACTAL_TOLERANCE: 0.2,         // D ∈ [1.2, 1.6] = биофильный диапазон
    ENTROPY_OPTIMAL_MIN: 4.0,       // Минимальная энтропия (бит) - не скучно
    ENTROPY_OPTIMAL_MAX: 6.0,       // Максимальная энтропия (бит) - не хаос
    ISOVIST_EYE_HEIGHT: 1.5,        // Высота глаз для расчета изовистов (м)
} as const;

/**
 * Утилита для расчета индекса вокселя в линейном массиве
 */
export function voxelIndex(i: number, j: number, k: number, nx: number, ny: number): number {
    return i + j * nx + k * nx * ny;
}

/**
 * Утилита для обратного преобразования индекса в координаты
 */
export function indexToCoords(index: number, nx: number, ny: number): [number, number, number] {
    const k = Math.floor(index / (nx * ny));
    const remainder = index % (nx * ny);
    const j = Math.floor(remainder / nx);
    const i = remainder % nx;
    return [i, j, k];
}
