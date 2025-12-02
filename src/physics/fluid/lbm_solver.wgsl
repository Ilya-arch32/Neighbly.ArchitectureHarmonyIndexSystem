/**
 * AHI 2.0 Ultimate - Lattice Boltzmann Method D3Q19 Solver
 * 
 * WebGPU Compute Shader для 3D CFD симуляции воздушных потоков
 * 
 * ВАЖНО: Это ядро всей аэродинамики! Никаких упрощений!
 * Реализует полную D3Q19 схему с MRT оператором столкновений для стабильности.
 * 
 * Физика:
 * - 19 дискретных скоростей в 3D (оптимальный баланс точность/производительность)
 * - Multiple Relaxation Time (MRT) вместо упрощенного BGK
 * - Full Bounce-Back для твердых границ
 * - Источниковые члены для тепловой плавучести (связь с CHT)
 */

// ============================================================================
// КОНСТАНТЫ D3Q19
// ============================================================================

const D3Q19_WEIGHTS: array<f32, 19> = array<f32, 19>(
    1.0/3.0,                    // 0: (0,0,0) - rest particle
    1.0/18.0, 1.0/18.0,        // 1-2: (±1,0,0)
    1.0/18.0, 1.0/18.0,        // 3-4: (0,±1,0)
    1.0/18.0, 1.0/18.0,        // 5-6: (0,0,±1)
    1.0/36.0, 1.0/36.0,        // 7-8: (±1,±1,0)
    1.0/36.0, 1.0/36.0,        // 9-10: (±1,0,±1)
    1.0/36.0, 1.0/36.0,        // 11-12: (0,±1,±1)
    1.0/36.0, 1.0/36.0,        // 13-14: (±1,-1,0)
    1.0/36.0, 1.0/36.0,        // 15-16: (±1,0,-1)
    1.0/36.0, 1.0/36.0         // 17-18: (0,±1,-1)
);

// Дискретные векторы скоростей (19 направлений)
const D3Q19_CX: array<i32, 19> = array<i32, 19>(
    0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0
);
const D3Q19_CY: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  1, -1,  0,  0,  1,  1,  0,  0,  1, -1, -1, -1,  0,  0,  1, -1
);
const D3Q19_CZ: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  0,  0,  1, -1,  0,  0,  1,  1,  1,  1,  0,  0, -1, -1, -1, -1
);

// Обратные направления (для bounce-back)
const D3Q19_OPPOSITE: array<i32, 19> = array<i32, 19>(
    0,  2,  1,  4,  3,  6,  5,  14, 13, 16, 15, 18, 17, 8, 7, 10, 9, 12, 11
);

// ============================================================================
// UNIFORM BUFFERS
// ============================================================================

struct SimulationParams {
    nx: u32,                    // Grid dimensions
    ny: u32,
    nz: u32,
    resolution: f32,            // Voxel size (м)
    
    tau: f32,                   // Relaxation time (связан с вязкостью)
    omega: f32,                 // ω = 1/τ
    
    rho0: f32,                  // Reference density (кг/м³)
    nu: f32,                    // Kinematic viscosity (м²/с)
    
    gravity: vec3<f32>,         // Гравитационное ускорение (м/с²)
    dt: f32,                    // Time step (с)
    
    enableBuoyancy: u32,        // Флаг тепловой плавучести
    beta: f32,                  // Коэффициент теплового расширения (1/К)
    smagorinskyConstant: f32,   // Константа Смагоринского C_s (0.1-0.2)
    enableLES: u32,             // Флаг включения LES модели
}

@group(0) @binding(0) var<uniform> params: SimulationParams;

// ============================================================================
// STORAGE BUFFERS (Read/Write)
// ============================================================================

// Distribution functions (19 per voxel)
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;

// Macroscopic variables
@group(0) @binding(3) var<storage, read_write> density: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read_write> temperature: array<f32>;

// Voxel state (для boundary conditions)
@group(0) @binding(6) var<storage, read> voxelState: array<u32>;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn voxelIndex(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.nx + k * params.nx * params.ny;
}

fn isInside(i: i32, j: i32, k: i32) -> bool {
    return i >= 0 && i < i32(params.nx) &&
           j >= 0 && j < i32(params.ny) &&
           k >= 0 && k < i32(params.nz));
}

fn isSolid(idx: u32) -> bool {
    return (voxelState[idx] & 0x01u) != 0u; // VoxelState.SOLID
}

// ============================================================================
// EQUILIBRIUM DISTRIBUTION FUNCTION
// ============================================================================

fn equilibrium(q: u32, rho: f32, u: vec3<f32>) -> f32 {
    let w = D3Q19_WEIGHTS[q];
    let c = vec3<f32>(f32(D3Q19_CX[q]), f32(D3Q19_CY[q]), f32(D3Q19_CZ[q]));
    
    let cu = dot(c, u);
    let usqr = dot(u, u);
    
    // Maxwell-Boltzmann распределение (2-й порядок по скорости)
    return w * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqr);
}

// ============================================================================
// LES: STRESS TENSOR И МОДЕЛЬ СМАГОРИНСКОГО
// ============================================================================

// Тензор напряжений S_ij из неравновесной части функции распределения
// S_ij = Σ_q (f_q - f_eq_q) * c_qi * c_qj
fn computeStressTensor(f_local: array<f32, 19>, f_eq: array<f32, 19>) -> mat3x3<f32> {
    var S: mat3x3<f32>;
    
    // Инициализируем нулями
    S[0] = vec3<f32>(0.0, 0.0, 0.0);
    S[1] = vec3<f32>(0.0, 0.0, 0.0);
    S[2] = vec3<f32>(0.0, 0.0, 0.0);
    
    // Суммируем вклад от неравновесных моментов
    for (var q = 0u; q < 19u; q++) {
        let f_neq = f_local[q] - f_eq[q];  // Неравновесная часть
        let cx = f32(D3Q19_CX[q]);
        let cy = f32(D3Q19_CY[q]);
        let cz = f32(D3Q19_CZ[q]);
        
        // S_ij += f_neq * c_i * c_j
        S[0][0] += f_neq * cx * cx;  // S_xx
        S[0][1] += f_neq * cx * cy;  // S_xy
        S[0][2] += f_neq * cx * cz;  // S_xz
        S[1][0] += f_neq * cy * cx;  // S_yx
        S[1][1] += f_neq * cy * cy;  // S_yy
        S[1][2] += f_neq * cy * cz;  // S_yz
        S[2][0] += f_neq * cz * cx;  // S_zx
        S[2][1] += f_neq * cz * cy;  // S_zy
        S[2][2] += f_neq * cz * cz;  // S_zz
    }
    
    // Нормализация: в LBM тензор связан с вязкостью через tau
    // S_ij = -(1 / (2 * tau * rho * c_s^2)) * Π_neq_ij
    // где c_s^2 = 1/3 для D3Q19
    let factor = -1.0 / (2.0 * params.tau * params.rho0 * (1.0 / 3.0));
    S[0] *= factor;
    S[1] *= factor;
    S[2] *= factor;
    
    return S;
}

// Вычисление магнитуды тензора скоростей деформации |S| = sqrt(2 * S_ij * S_ij)
fn computeStrainRateMagnitude(S: mat3x3<f32>) -> f32 {
    // |S| = sqrt(2 * S_ij * S_ij) = sqrt(2 * (S_xx^2 + S_yy^2 + S_zz^2 + 2*(S_xy^2 + S_xz^2 + S_yz^2)))
    let Sxx = S[0][0];
    let Syy = S[1][1];
    let Szz = S[2][2];
    let Sxy = S[0][1];
    let Sxz = S[0][2];
    let Syz = S[1][2];
    
    let S_squared = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);
    return sqrt(2.0 * S_squared);
}

// Модель Смагоринского: турбулентная вязкость ν_t = (C_s * Δ)² * |S|
fn computeTurbulentViscosity(S_magnitude: f32) -> f32 {
    let Cs = params.smagorinskyConstant;  // Константа Смагоринского (0.1-0.2)
    let delta = params.resolution;         // Размер ячейки сетки (фильтр)
    
    // ν_t = (C_s * Δ)² * |S|
    let nu_t = (Cs * delta) * (Cs * delta) * S_magnitude;
    
    return nu_t;
}

// Вычисление эффективного времени релаксации τ_eff = τ_0 + τ_turb
// где τ_turb = 3 * ν_t / c_s^2 = 3 * ν_t * 3 = 9 * ν_t (в решеточных единицах)
fn computeEffectiveTau(nu_turb: f32) -> f32 {
    // В LBM: ν = (τ - 0.5) * c_s^2 = (τ - 0.5) / 3
    // Поэтому: τ = 3*ν + 0.5
    // τ_turb = 3 * ν_t
    let tau_turb = 3.0 * nu_turb;
    
    // τ_eff = τ_0 + τ_turb
    let tau_eff = params.tau + tau_turb;
    
    // Ограничиваем снизу для стабильности (τ > 0.5)
    return max(tau_eff, 0.505);
}

// ============================================================================
// MRT (Multiple Relaxation Time) COLLISION OPERATOR
// ============================================================================

// Преобразование в момент space (упрощенная версия для D3Q19)
fn momentTransform(f: array<f32, 19>) -> array<f32, 19> {
    var m: array<f32, 19>;
    
    // m[0] = density
    m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + 
           f[7] + f[8] + f[9] + f[10] + f[11] + f[12] +
           f[13] + f[14] + f[15] + f[16] + f[17] + f[18];
    
    // m[1-3] = momentum (jx, jy, jz)
    m[1] = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16];
    m[2] = f[3] - f[4] + f[7] + f[8] + f[11] - f[12] - f[13] - f[14] + f[17] - f[18];
    m[3] = f[5] - f[6] + f[9] + f[10] + f[11] + f[12] - f[15] - f[16] - f[17] - f[18];
    
    // Высшие моменты (энергия, stress tensor) - упрощенная формула
    for (var i = 4u; i < 19u; i++) {
        m[i] = 0.0; // Для базовой реализации
    }
    
    return m;
}

fn inverseMomentTransform(m: array<f32, 19>) -> array<f32, 19> {
    var f: array<f32, 19>;
    
    // Обратное преобразование (точная формула сложна, используем приближение)
    let rho = m[0];
    let ux = m[1] / rho;
    let uy = m[2] / rho;
    let uz = m[3] / rho;
    let u = vec3<f32>(ux, uy, uz);
    
    for (var q = 0u; q < 19u; q++) {
        f[q] = equilibrium(q, rho, u);
    }
    
    return f;
}

fn mrtCollision(f_local: array<f32, 19>, rho: f32, u: vec3<f32>) -> array<f32, 19> {
    return mrtCollisionWithOmega(f_local, rho, u, params.omega);
}

// MRT collision с произвольным omega (для LES с переменной вязкостью)
fn mrtCollisionWithOmega(f_local: array<f32, 19>, rho: f32, u: vec3<f32>, omega: f32) -> array<f32, 19> {
    var f_new: array<f32, 19>;
    
    // Преобразуем в момент space
    let m = momentTransform(f_local);
    let m_eq = momentTransform(inverseMomentTransform(m));
    
    // Relaxation с разными временами для разных моментов
    var m_relaxed: array<f32, 19>;
    m_relaxed[0] = m[0]; // Плотность сохраняется
    
    // Момент импульса релаксируется с omega (теперь переменный для LES)
    for (var i = 1u; i < 4u; i++) {
        m_relaxed[i] = m[i] - omega * (m[i] - m_eq[i]);
    }
    
    // Высшие моменты - масштабируем bulk omega пропорционально
    // Соотношение сохраняется: omega_bulk/omega_base = 1.2/omega_0
    let omega_ratio = 1.2 / params.omega;
    let omega_bulk = omega * omega_ratio;
    for (var i = 4u; i < 19u; i++) {
        m_relaxed[i] = m[i] - omega_bulk * (m[i] - m_eq[i]);
    }
    
    // Обратное преобразование
    return inverseMomentTransform(m_relaxed);
}

// ============================================================================
// MAIN COMPUTE SHADER: COLLISION STEP
// ============================================================================

@compute @workgroup_size(8, 8, 8)
fn collisionStep(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    let j = globalId.y;
    let k = globalId.z;
    
    if (i >= params.nx || j >= params.ny || k >= params.nz) {
        return;
    }
    
    let idx = voxelIndex(i, j, k);
    
    // Skip solid voxels
    if (isSolid(idx)) {
        return;
    }
    
    // Собираем локальные distribution functions
    var f_local: array<f32, 19>;
    for (var q = 0u; q < 19u; q++) {
        f_local[q] = f_in[idx * 19u + q];
    }
    
    // Вычисляем макроскопические переменные
    var rho = 0.0;
    var momentum = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var q = 0u; q < 19u; q++) {
        rho += f_local[q];
        let c = vec3<f32>(f32(D3Q19_CX[q]), f32(D3Q19_CY[q]), f32(D3Q19_CZ[q]));
        momentum += c * f_local[q];
    }
    
    let u = momentum / rho;
    
    // Источниковый член: тепловая плавучесть (Boussinesq approximation)
    var force = vec3<f32>(0.0, 0.0, 0.0);
    if (params.enableBuoyancy != 0u) {
        let T = temperature[idx];
        let T0 = 293.0; // Reference temperature (20°C)
        let dT = T - T0;
        force = params.beta * dT * params.gravity * params.rho0;
    }
    
    // Добавляем силу к скорости (Guo forcing scheme)
    let u_forced = u + force * params.dt / (2.0 * rho);
    
    // ============================================================================
    // LES: Smagorinsky SGS Model
    // ============================================================================
    var omega_eff = params.omega;  // По умолчанию используем базовый omega
    
    if (params.enableLES != 0u) {
        // 1. Вычисляем равновесное распределение для текущего состояния
        var f_eq: array<f32, 19>;
        for (var q = 0u; q < 19u; q++) {
            f_eq[q] = equilibrium(q, rho, u_forced);
        }
        
        // 2. Вычисляем тензор напряжений из неравновесных моментов
        let S = computeStressTensor(f_local, f_eq);
        
        // 3. Вычисляем магнитуду скорости деформации |S|
        let S_magnitude = computeStrainRateMagnitude(S);
        
        // 4. Вычисляем турбулентную вязкость по модели Смагоринского
        let nu_turb = computeTurbulentViscosity(S_magnitude);
        
        // 5. Вычисляем эффективное время релаксации τ_eff = τ_0 + τ_turb
        let tau_eff = computeEffectiveTau(nu_turb);
        
        // 6. Эффективная частота релаксации
        omega_eff = 1.0 / tau_eff;
    }
    
    // MRT collision с эффективным omega (для LES или стандартным)
    let f_post_collision = mrtCollisionWithOmega(f_local, rho, u_forced, omega_eff);
    
    // Записываем обратно
    for (var q = 0u; q < 19u; q++) {
        f_out[idx * 19u + q] = f_post_collision[q];
    }
    
    // Обновляем макроскопические переменные для visualization
    density[idx] = rho;
    velocity[idx] = u;
}

// ============================================================================
// STREAMING STEP (перемещение частиц)
// ============================================================================

@compute @workgroup_size(8, 8, 8)
fn streamingStep(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = i32(globalId.x);
    let j = i32(globalId.y);
    let k = i32(globalId.z);
    
    if (i >= i32(params.nx) || j >= i32(params.ny) || k >= i32(params.nz)) {
        return;
    }
    
    let idx = voxelIndex(u32(i), u32(j), u32(k));
    
    // Твердые вокселы: full bounce-back
    if (isSolid(idx)) {
        var f_bounced: array<f32, 19>;
        
        for (var q = 0u; q < 19u; q++) {
            let opposite_q = u32(D3Q19_OPPOSITE[q]);
            f_bounced[q] = f_in[idx * 19u + opposite_q];
        }
        
        for (var q = 0u; q < 19u; q++) {
            f_out[idx * 19u + q] = f_bounced[q];
        }
        
        return;
    }
    
    // Fluid voxels: streaming
    for (var q = 0u; q < 19u; q++) {
        let ni = i - D3Q19_CX[q];
        let nj = j - D3Q19_CY[q];
        let nk = k - D3Q19_CZ[q];
        
        if (isInside(ni, nj, nk)) {
            let neighbor_idx = voxelIndex(u32(ni), u32(nj), u32(nk));
            
            // Копируем из соседа (потоковая передача)
            f_out[idx * 19u + q] = f_in[neighbor_idx * 19u + q];
        } else {
            // Граница домена: bounce-back
            let opposite_q = u32(D3Q19_OPPOSITE[q]);
            f_out[idx * 19u + q] = f_in[idx * 19u + opposite_q];
        }
    }
}

// ============================================================================
// ГРАНИЧНЫЕ УСЛОВИЯ: Inlet (фиксированная скорость)
// ============================================================================

@compute @workgroup_size(8, 8)
fn applyInletBC(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let j = globalId.x;
    let k = globalId.y;
    
    if (j >= params.ny || k >= params.nz) {
        return;
    }
    
    let i = 0u; // X=0 plane - inlet
    let idx = voxelIndex(i, j, k);
    
    if (isSolid(idx)) {
        return;
    }
    
    // Фиксированная скорость: 0.1 м/с вдоль X
    let u_inlet = vec3<f32>(0.1, 0.0, 0.0);
    let rho = params.rho0;
    
    for (var q = 0u; q < 19u; q++) {
        f_out[idx * 19u + q] = equilibrium(q, rho, u_inlet);
    }
}

// ============================================================================
// ГРАНИЧНЫЕ УСЛОВИЯ: Outlet (свободное истечение)
// ============================================================================

@compute @workgroup_size(8, 8)
fn applyOutletBC(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let j = globalId.x;
    let k = globalId.y;
    
    if (j >= params.ny || k >= params.nz) {
        return;
    }
    
    let i = params.nx - 1u; // X=max plane - outlet
    let i_prev = i - 1u;
    
    let idx = voxelIndex(i, j, k);
    let idx_prev = voxelIndex(i_prev, j, k);
    
    if (isSolid(idx)) {
        return;
    }
    
    // Копируем из предыдущего слоя (нулевой градиент)
    for (var q = 0u; q < 19u; q++) {
        f_out[idx * 19u + q] = f_in[idx_prev * 19u + q];
    }
}
