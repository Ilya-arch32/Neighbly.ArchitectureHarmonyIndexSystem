/**
 * AHI 2.0 Ultimate - Conjugate Heat Transfer (CHT) Solver
 * 
 * Решает уравнение теплопроводности в твердых телах и конвективный теплообмен в жидкостях.
 * Связывает температуру стен с температурой воздуха через граничные условия.
 */

struct Uniforms {
    grid_size: vec3<u32>,
    resolution: f32,
    dt: f32,
    alpha_solid: f32,    // Thermal diffusivity твердого тела [m²/s]
    alpha_fluid: f32,    // Thermal diffusivity воздуха [m²/s]
    h_conv: f32,         // Heat transfer coefficient [W/(m²·K)]
    gravity: vec3<f32>,
    beta: f32,           // Thermal expansion coefficient [1/K]
    T_ref: f32,          // Reference temperature [K]
    
    // ISO 13788 параметры влагопереноса
    D_v: f32,            // Коэффициент диффузии пара [m²/s]
    moldRiskThreshold: f32, // Порог RH для риска плесени (0.8 = 80%)
    moldRiskSteps: u32,  // Количество шагов для фиксации риска
}

// Voxel states (matching VoxelTypes.ts)
const VOXEL_EMPTY: u32 = 0u;
const VOXEL_SOLID: u32 = 1u;
const VOXEL_FLUID: u32 = 2u;
const VOXEL_GLASS: u32 = 4u;
const VOXEL_BOUNDARY: u32 = 8u;
const VOXEL_MOLD_RISK: u32 = 256u; // 0x100 - Флаг риска плесени (ISO 13788)

// Material properties lookup
fn get_thermal_conductivity(material_id: u32) -> f32 {
    // Hardcoded for now, should be in buffer
    switch material_id {
        case 0u: { return 0.026; }  // Air
        case 1u: { return 1.4; }    // Concrete
        case 2u: { return 0.15; }   // Wood
        case 3u: { return 1.0; }    // Glass
        default: { return 1.0; }
    }
}

fn get_specific_heat(material_id: u32) -> f32 {
    switch material_id {
        case 0u: { return 1005.0; }  // Air
        case 1u: { return 880.0; }   // Concrete
        case 2u: { return 1700.0; }  // Wood
        case 3u: { return 840.0; }   // Glass
        default: { return 1000.0; }
    }
}

fn get_density(material_id: u32) -> f32 {
    switch material_id {
        case 0u: { return 1.225; }   // Air
        case 1u: { return 2400.0; }  // Concrete
        case 2u: { return 600.0; }   // Wood
        case 3u: { return 2500.0; }  // Glass
        default: { return 1000.0; }
    }
}

// Паропроницаемость материалов (ISO 13788)
// μ - коэффициент сопротивления паропроницанию
fn get_vapor_permeability(material_id: u32) -> f32 {
    switch material_id {
        case 0u: { return 1.0; }     // Air (μ=1)
        case 1u: { return 100.0; }   // Concrete (μ~50-200)
        case 2u: { return 20.0; }    // Wood (μ~5-50)
        case 3u: { return 1000000.0; } // Glass (практически непроницаем)
        case 4u: { return 5.0; }     // Insulation
        case 5u: { return 15.0; }    // Brick
        default: { return 50.0; }
    }
}

// ============================================================================
// MAGNUS FORMULA - Давление насыщенного пара (ISO 13788)
// ============================================================================

// Формула Магнуса для расчета давления насыщенного водяного пара
// p_sat(T) = 610.94 * exp(17.625 * T_c / (T_c + 243.04)) [Pa]
// где T_c - температура в градусах Цельсия
fn saturated_vapor_pressure(T_kelvin: f32) -> f32 {
    let T_celsius = T_kelvin - 273.15;
    
    // Ограничиваем диапазон для стабильности
    let T_c = clamp(T_celsius, -40.0, 80.0);
    
    // Magnus formula constants (Buck, 1981)
    let a = 17.625;
    let b = 243.04;
    
    return 610.94 * exp(a * T_c / (T_c + b));
}

// Расчет точки росы (Dew Point) по формуле Магнуса
// T_dew = b * [α(T,RH) / (a - α(T,RH))]
// где α(T,RH) = a*T_c/(b+T_c) + ln(RH)
fn dew_point_temperature(T_kelvin: f32, RH: f32) -> f32 {
    let T_celsius = T_kelvin - 273.15;
    let T_c = clamp(T_celsius, -40.0, 80.0);
    let rh = clamp(RH, 0.01, 1.0); // Избегаем ln(0)
    
    let a = 17.625;
    let b = 243.04;
    
    let alpha = a * T_c / (b + T_c) + log(rh);
    let T_dew_c = b * alpha / (a - alpha);
    
    return T_dew_c + 273.15; // Возвращаем в Кельвинах
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> voxel_state: array<f32>; // [state, material, ...]
@group(0) @binding(2) var<storage, read> temperature_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> temperature_out: array<f32>;
@group(0) @binding(4) var<storage, read> velocity: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read_write> heat_flux: array<f32>;

fn index_1d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * uniforms.grid_size.x + z * uniforms.grid_size.x * uniforms.grid_size.y;
}

fn is_boundary(x: u32, y: u32, z: u32) -> bool {
    // Check if this voxel is at interface between solid and fluid
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    
    if state != VOXEL_SOLID && state != VOXEL_FLUID {
        return false;
    }
    
    // Check neighbors
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 16u]);
            
            // Boundary if solid next to fluid or vice versa
            if ((state == VOXEL_SOLID && n_state == VOXEL_FLUID) ||
                (state == VOXEL_FLUID && n_state == VOXEL_SOLID)) {
                return true;
            }
        }
    }
    
    return false;
}

/**
 * Heat Diffusion in Solids (Fourier's Law)
 */
@compute @workgroup_size(8, 8, 8)
fn diffusion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    let material = u32(voxel_state[idx * 16u + 1u]);
    
    // Only process solid voxels
    if (state != VOXEL_SOLID) {
        temperature_out[idx] = temperature_in[idx];
        return;
    }
    
    let T_center = temperature_in[idx];
    let k = get_thermal_conductivity(material);
    let rho = get_density(material);
    let cp = get_specific_heat(material);
    let alpha = k / (rho * cp); // Thermal diffusivity
    
    // Finite difference for heat equation: ∂T/∂t = α∇²T
    var laplacian = 0.0;
    var count = 0u;
    
    // 6-point stencil
    if (x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        if (u32(voxel_state[idx_m * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        }
    }
    if (x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        if (u32(voxel_state[idx_p * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        }
    }
    
    if (y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        if (u32(voxel_state[idx_m * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        }
    }
    if (y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        if (u32(voxel_state[idx_p * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        }
    }
    
    if (z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        if (u32(voxel_state[idx_m * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        }
    }
    if (z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        if (u32(voxel_state[idx_p * 16u]) == VOXEL_SOLID) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        }
    }
    
    if (count > 0u) {
        laplacian = laplacian / (uniforms.resolution * uniforms.resolution);
        temperature_out[idx] = T_center + alpha * uniforms.dt * laplacian;
    } else {
        temperature_out[idx] = T_center;
    }
}

/**
 * Convective Heat Transfer in Fluids
 */
@compute @workgroup_size(8, 8, 8)
fn convection_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    
    // Only process fluid voxels
    if (state != VOXEL_FLUID) {
        return;
    }
    
    let T_center = temperature_in[idx];
    let v = velocity[idx];
    
    // Upwind advection scheme
    var dTdx = 0.0;
    var dTdy = 0.0;
    var dTdz = 0.0;
    
    // X direction
    if (v.x > 0.0 && x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        dTdx = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.x < 0.0 && x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        dTdx = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Y direction
    if (v.y > 0.0 && y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        dTdy = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.y < 0.0 && y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        dTdy = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Z direction
    if (v.z > 0.0 && z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        dTdz = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.z < 0.0 && z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        dTdz = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Advection term: -v·∇T
    let advection = -(v.x * dTdx + v.y * dTdy + v.z * dTdz);
    
    // Diffusion (simplified)
    let alpha_air = uniforms.alpha_fluid;
    var laplacian = 0.0;
    var count = 0u;
    
    // Check all 6 neighbors for diffusion
    if (x > 0u) {
        laplacian += temperature_in[index_1d(x - 1u, y, z)] - T_center;
        count += 1u;
    }
    if (x < uniforms.grid_size.x - 1u) {
        laplacian += temperature_in[index_1d(x + 1u, y, z)] - T_center;
        count += 1u;
    }
    if (y > 0u) {
        laplacian += temperature_in[index_1d(x, y - 1u, z)] - T_center;
        count += 1u;
    }
    if (y < uniforms.grid_size.y - 1u) {
        laplacian += temperature_in[index_1d(x, y + 1u, z)] - T_center;
        count += 1u;
    }
    if (z > 0u) {
        laplacian += temperature_in[index_1d(x, y, z - 1u)] - T_center;
        count += 1u;
    }
    if (z < uniforms.grid_size.z - 1u) {
        laplacian += temperature_in[index_1d(x, y, z + 1u)] - T_center;
        count += 1u;
    }
    
    if (count > 0u) {
        laplacian = laplacian / (uniforms.resolution * uniforms.resolution);
    }
    
    // Update temperature: ∂T/∂t = -v·∇T + α∇²T
    temperature_out[idx] = T_center + uniforms.dt * (advection + alpha_air * laplacian);
}

/**
 * Boundary Coupling: Heat exchange between solid and fluid
 */
@compute @workgroup_size(8, 8, 8)
fn boundary_coupling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    // Only process boundary voxels
    if (!is_boundary(x, y, z)) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    
    // Find adjacent solid/fluid pairs
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    var total_flux = 0.0;
    var flux_count = 0u;
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 16u]);
            
            // Heat transfer at solid-fluid interface
            if ((state == VOXEL_SOLID && n_state == VOXEL_FLUID) ||
                (state == VOXEL_FLUID && n_state == VOXEL_SOLID)) {
                
                let T_solid = select(temperature_in[idx], temperature_in[n_idx], state == VOXEL_FLUID);
                let T_fluid = select(temperature_in[n_idx], temperature_in[idx], state == VOXEL_FLUID);
                
                // Newton's law of cooling: q = h * A * (T_wall - T_fluid)
                // Adjust h based on local velocity (forced convection)
                let v_local = select(velocity[n_idx], velocity[idx], state == VOXEL_FLUID);
                let v_mag = length(v_local);
                
                // Enhanced heat transfer with flow (simplified correlation)
                let h_effective = uniforms.h_conv * (1.0 + 0.5 * sqrt(v_mag));
                
                let area = uniforms.resolution * uniforms.resolution; // Face area
                let q = h_effective * area * (T_solid - T_fluid);
                
                total_flux += q;
                flux_count += 1u;
            }
        }
    }
    
    if (flux_count > 0u) {
        heat_flux[idx] = total_flux / f32(flux_count);
        
        // Apply flux to temperature
        let material = u32(voxel_state[idx * 16u + 1u]);
        let rho = get_density(material);
        let cp = get_specific_heat(material);
        let volume = uniforms.resolution * uniforms.resolution * uniforms.resolution;
        
        // ∆T = Q / (m * cp) = q * dt / (ρ * V * cp)
        let dT = heat_flux[idx] * uniforms.dt / (rho * volume * cp);
        
        if (state == VOXEL_SOLID) {
            temperature_out[idx] -= dT; // Solid loses heat
        } else {
            temperature_out[idx] += dT; // Fluid gains heat
        }
    }
}

/**
 * Compute Buoyancy Force for LBM
 * Returns force vector based on temperature difference
 */
@compute @workgroup_size(8, 8, 8)
fn compute_buoyancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    
    // Only compute for fluid
    if (state != VOXEL_FLUID) {
        heat_flux[idx] = 0.0; // Reuse buffer for buoyancy magnitude
        return;
    }
    
    let T = temperature_out[idx];
    
    // Boussinesq approximation: F_buoyancy = -ρ₀ * g * β * (T - T_ref)
    // Direction is opposite to gravity
    let buoyancy_magnitude = uniforms.beta * abs(T - uniforms.T_ref);
    
    // Store magnitude for LBM to use
    heat_flux[idx] = buoyancy_magnitude;
}

// ============================================================================
// ISO 13788: ДИФФУЗИЯ ВОДЯНОГО ПАРА
// ============================================================================

// Буферы влажности (добавляются через bind group)
@group(1) @binding(0) var<storage, read> humidity_in: array<f32>;
@group(1) @binding(1) var<storage, read_write> humidity_out: array<f32>;
@group(1) @binding(2) var<storage, read_write> mold_risk: array<u32>;
@group(1) @binding(3) var<storage, read_write> mold_risk_counter: array<u32>;

/**
 * Диффузия водяного пара (ISO 13788)
 * 
 * Уравнение диффузии пара аналогично диффузии тепла:
 * ∂φ/∂t = D_eff * ∇²φ
 * 
 * где D_eff = D_v / μ (эффективный коэффициент диффузии с учетом материала)
 * φ - относительная влажность (0.0 - 1.0)
 */
@compute @workgroup_size(8, 8, 8)
fn vapor_diffusion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    let material = u32(voxel_state[idx * 16u + 1u]);
    
    // Пропускаем пустые воксели
    if (state == VOXEL_EMPTY) {
        humidity_out[idx] = humidity_in[idx];
        return;
    }
    
    let phi_center = humidity_in[idx];
    let mu = get_vapor_permeability(material);
    
    // Эффективный коэффициент диффузии
    let D_eff = uniforms.D_v / mu;
    
    // Лапласиан влажности (6-точечный стенсил)
    var laplacian = 0.0;
    var count = 0u;
    let h2 = uniforms.resolution * uniforms.resolution;
    
    // X direction
    if (x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        let state_m = u32(voxel_state[idx_m * 16u]);
        if (state_m != VOXEL_EMPTY) {
            // Учитываем разную паропроницаемость на границе
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m); // Гармоническое среднее
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        let state_p = u32(voxel_state[idx_p * 16u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Y direction
    if (y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        let state_m = u32(voxel_state[idx_m * 16u]);
        if (state_m != VOXEL_EMPTY) {
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m);
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        let state_p = u32(voxel_state[idx_p * 16u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Z direction
    if (z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        let state_m = u32(voxel_state[idx_m * 16u]);
        if (state_m != VOXEL_EMPTY) {
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m);
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        let state_p = u32(voxel_state[idx_p * 16u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 16u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Обновляем влажность
    if (count > 0u) {
        let new_phi = phi_center + uniforms.dt * laplacian;
        // Ограничиваем в физических пределах [0, 1]
        humidity_out[idx] = clamp(new_phi, 0.0, 1.0);
    } else {
        humidity_out[idx] = phi_center;
    }
}

/**
 * Оценка риска плесени (ISO 13788)
 * 
 * Условия для MOLD_RISK:
 * 1. Температура поверхности < точки росы (T_surface < T_dew)
 * 2. ИЛИ относительная влажность > 80% в течение X шагов
 */
@compute @workgroup_size(8, 8, 8)
fn calculate_mold_risk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 16u]);
    
    // Проверяем только граничные воксели (твердые рядом с воздухом)
    if (state != VOXEL_SOLID) {
        mold_risk[idx] = 0u;
        return;
    }
    
    // Проверяем, является ли это поверхностью
    var is_surface = false;
    var adjacent_fluid_idx = idx;
    
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 16u]);
            
            if (n_state == VOXEL_FLUID) {
                is_surface = true;
                adjacent_fluid_idx = n_idx;
                break;
            }
        }
    }
    
    if (!is_surface) {
        mold_risk[idx] = 0u;
        return;
    }
    
    // Получаем температуру поверхности и влажность воздуха
    let T_surface = temperature_out[idx];
    let T_air = temperature_out[adjacent_fluid_idx];
    let RH_air = humidity_out[adjacent_fluid_idx];
    
    // Расчет точки росы
    let T_dew = dew_point_temperature(T_air, RH_air);
    
    // Расчет относительной влажности на поверхности
    // RH_surface = p_v_air / p_sat(T_surface)
    // p_v_air = RH_air * p_sat(T_air)
    let p_sat_air = saturated_vapor_pressure(T_air);
    let p_v_air = RH_air * p_sat_air;
    let p_sat_surface = saturated_vapor_pressure(T_surface);
    let RH_surface = p_v_air / p_sat_surface;
    
    // Проверяем условия риска плесени
    var risk_condition = false;
    
    // Условие 1: Температура поверхности ниже точки росы (конденсация)
    if (T_surface < T_dew) {
        risk_condition = true;
    }
    
    // Условие 2: Относительная влажность на поверхности > порога
    if (RH_surface > uniforms.moldRiskThreshold) {
        risk_condition = true;
    }
    
    // Обновляем счетчик
    var counter = mold_risk_counter[idx];
    
    if (risk_condition) {
        counter += 1u;
    } else {
        // Медленно сбрасываем счетчик при улучшении условий
        if (counter > 0u) {
            counter -= 1u;
        }
    }
    
    mold_risk_counter[idx] = counter;
    
    // Помечаем флагом MOLD_RISK если счетчик превышает порог
    if (counter >= uniforms.moldRiskSteps) {
        mold_risk[idx] = VOXEL_MOLD_RISK;
    } else {
        mold_risk[idx] = 0u;
    }
}
