"""
RC-Network Thermal Model Implementation
Based on ISO 13790 simplified hourly method (5R1C model)

This model represents building thermal dynamics as an electrical circuit:
- Resistors (R) = thermal resistance of materials
- Capacitors (C) = thermal mass/heat storage capacity
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class ThermalNode:
    """Represents a thermal node in the RC network"""
    temperature: float  # Current temperature [°C]
    capacitance: float  # Thermal capacitance [J/K]
    name: str


@dataclass
class ThermalResistance:
    """Represents thermal resistance between two nodes"""
    value: float  # Thermal resistance [K/W]
    node1: str
    node2: str


class RCNetworkModel:
    """
    5R1C model implementation for dynamic thermal simulation
    
    The model consists of:
    - 5 Resistances: 
        * R_si: Internal surface resistance
        * R_ms: Resistance of massive part
        * R_em: External part of wall
        * R_window: Window resistance
        * R_ventilation: Ventilation/infiltration
    - 1 Capacitance:
        * C_m: Thermal mass of the building
    """
    
    def __init__(self, building_params: Dict):
        """
        Initialize RC Network Model
        
        Args:
            building_params: Dictionary containing:
                - wall_area: Total wall area [m²]
                - window_area: Total window area [m²]
                - floor_area: Floor area [m²]
                - volume: Room volume [m³]
                - wall_thickness: Wall thickness [m]
                - wall_material: Material properties dict
                - u_value_wall: Wall U-value [W/m²K]
                - u_value_window: Window U-value [W/m²K]
                - air_change_rate: Ventilation rate [1/h]
        """
        self.params = building_params
        self._setup_network()
        
    def _setup_network(self):
        """Setup the RC network components"""
        
        # Extract parameters
        A_wall = self.params['wall_area']
        A_window = self.params['window_area']
        A_floor = self.params['floor_area']
        V = self.params['volume']
        d_wall = self.params['wall_thickness']
        
        # Material properties (default concrete if not specified)
        material = self.params.get('wall_material', {
            'density': 2400,  # kg/m³
            'specific_heat': 1000,  # J/kg·K
            'conductivity': 1.7  # W/m·K
        })
        
        # Calculate thermal resistances [K/W]
        self.R_si = 0.13 / (A_wall + A_window)  # Internal surface resistance
        self.R_window = 1.0 / (self.params['u_value_window'] * A_window)
        
        # Split wall resistance into massive and external parts
        R_wall_total = 1.0 / (self.params['u_value_wall'] * A_wall)
        self.R_ms = R_wall_total * 0.5  # Massive part (50%)
        self.R_em = R_wall_total * 0.5  # External part (50%)
        
        # Ventilation resistance
        n_air = self.params['air_change_rate']  # Air changes per hour
        self.R_ventilation = 1.0 / (0.34 * n_air * V / 3600)  # 0.34 Wh/m³K for air
        
        # Calculate thermal capacitance [J/K]
        # Effective thermal mass (considering only the active layer ~10cm)
        effective_thickness = min(d_wall, 0.1)  # Maximum 10cm active layer
        mass_wall = A_wall * effective_thickness * material['density']
        
        # Add internal thermal mass (furniture, air, etc.)
        mass_internal = A_floor * 50  # Approximate 50 kg/m² for furnished room
        
        self.C_m = (mass_wall * material['specific_heat'] + 
                   mass_internal * 500)  # 500 J/kg·K average for furniture
        
        # Initialize nodes
        self.nodes = {
            'interior': ThermalNode(20.0, 0, 'interior'),  # Air node (no capacity)
            'mass': ThermalNode(20.0, self.C_m, 'mass'),  # Thermal mass node
            'exterior': ThermalNode(0.0, 0, 'exterior')  # External temperature
        }
        
        # Store network structure
        self.resistances = [
            ThermalResistance(self.R_si, 'interior', 'mass'),
            ThermalResistance(self.R_ms, 'mass', 'exterior'),
            ThermalResistance(self.R_em, 'mass', 'exterior'),
            ThermalResistance(self.R_window, 'interior', 'exterior'),
            ThermalResistance(self.R_ventilation, 'interior', 'exterior')
        ]
        
    def _heat_flow(self, R: float, T1: float, T2: float) -> float:
        """Calculate heat flow through resistance"""
        return (T1 - T2) / R if R > 0 else 0
    
    def _derivatives(self, t: float, y: np.ndarray, 
                    T_ext: float, Q_solar: float, Q_internal: float) -> np.ndarray:
        """
        Calculate temperature derivatives for ODE solver
        
        Args:
            t: Time [s]
            y: State vector [T_interior, T_mass]
            T_ext: External temperature [°C]
            Q_solar: Solar heat gains [W]
            Q_internal: Internal heat gains [W]
        
        Returns:
            dy/dt: Derivatives [dT_interior/dt, dT_mass/dt]
        """
        T_int, T_mass = y
        
        # Heat flows [W]
        Q_si = self._heat_flow(self.R_si, T_int, T_mass)
        Q_ms = self._heat_flow(self.R_ms, T_mass, T_ext)
        Q_em = self._heat_flow(self.R_em, T_mass, T_ext)
        Q_window = self._heat_flow(self.R_window, T_int, T_ext)
        Q_vent = self._heat_flow(self.R_ventilation, T_int, T_ext)
        
        # Energy balance for interior air (no thermal mass)
        # Assuming quasi-steady state for air temperature
        # Q_in = Q_out => Find equilibrium temperature
        
        # Total heat loss from interior
        R_total_int = 1.0 / (1/self.R_si + 1/self.R_window + 1/self.R_ventilation)
        
        # Interior temperature in equilibrium
        T_int_eq = (T_mass/self.R_si + T_ext*(1/self.R_window + 1/self.R_ventilation) + 
                   (Q_solar + Q_internal)*R_total_int) * R_total_int
        
        # Fast response for air temperature (time constant ~ 1 minute)
        tau_air = 60  # seconds
        dT_int_dt = (T_int_eq - T_int) / tau_air
        
        # Energy balance for thermal mass
        dT_mass_dt = (Q_si - Q_ms - Q_em + 0.5*Q_solar) / self.C_m
        
        return np.array([dT_int_dt, dT_mass_dt])
    
    def simulate_period(self, weather_data: Dict, 
                       initial_temps: Tuple[float, float] = (20.0, 20.0),
                       internal_gains_schedule: List[float] = None) -> Dict:
        """
        Simulate thermal behavior over a time period
        
        Args:
            weather_data: Dictionary with hourly data:
                - temperature: List of external temperatures [°C]
                - solar_radiation: List of solar radiation [W/m²]
            initial_temps: Initial (interior, mass) temperatures
            internal_gains_schedule: Hourly internal gains [W]
        
        Returns:
            Dictionary with simulation results:
                - time: Time points [hours]
                - T_interior: Interior air temperature [°C]
                - T_mass: Thermal mass temperature [°C]
                - heating_demand: Required heating [W]
                - cooling_demand: Required cooling [W]
        """
        
        # Prepare data
        T_ext_data = np.array(weather_data['temperature'])
        solar_data = np.array(weather_data.get('solar_radiation', np.zeros_like(T_ext_data)))
        
        n_hours = len(T_ext_data)
        
        if internal_gains_schedule is None:
            # Default schedule: 5 W/m² during day, 2 W/m² at night
            internal_gains_schedule = []
            for hour in range(n_hours):
                h = hour % 24
                if 7 <= h <= 22:
                    internal_gains_schedule.append(5 * self.params['floor_area'])
                else:
                    internal_gains_schedule.append(2 * self.params['floor_area'])
        
        # Initialize results
        results = {
            'time': [],
            'T_interior': [],
            'T_mass': [],
            'T_exterior': [],
            'heating_demand': [],
            'cooling_demand': [],
            'thermal_lag': []
        }
        
        # Initial conditions
        y0 = np.array(initial_temps)
        
        # Simulate hour by hour
        for hour in range(n_hours):
            # External conditions for this hour
            T_ext = T_ext_data[hour]
            
            # Solar gains (simplified: 50% of radiation on 30% of window area)
            Q_solar = 0.5 * 0.3 * solar_data[hour] * self.params['window_area']
            
            # Internal gains
            Q_internal = internal_gains_schedule[hour]
            
            # Solve ODE for this hour
            sol = solve_ivp(
                lambda t, y: self._derivatives(t, y, T_ext, Q_solar, Q_internal),
                t_span=[hour*3600, (hour+1)*3600],
                y0=y0,
                method='RK45',
                dense_output=True
            )
            
            # Extract final values
            y_final = sol.y[:, -1]
            T_int = y_final[0]
            T_mass = y_final[1]
            
            # Calculate heating/cooling demand to maintain comfort (20-26°C)
            heating = 0
            cooling = 0
            T_setpoint_heat = 20.0
            T_setpoint_cool = 26.0
            
            if T_int < T_setpoint_heat:
                # Heating needed
                heating = (T_setpoint_heat - T_int) / self.R_si * 1000  # Convert to W
            elif T_int > T_setpoint_cool:
                # Cooling needed  
                cooling = (T_int - T_setpoint_cool) / self.R_si * 1000
            
            # Calculate thermal lag (phase shift between exterior and interior)
            if hour > 0:
                dT_ext = T_ext - T_ext_data[hour-1]
                dT_int = T_int - results['T_interior'][-1] if results['T_interior'] else 0
                if abs(dT_ext) > 0.1:
                    lag = dT_int / dT_ext if dT_ext != 0 else 0
                else:
                    lag = results['thermal_lag'][-1] if results['thermal_lag'] else 0
            else:
                lag = 0
            
            # Store results
            results['time'].append(hour)
            results['T_interior'].append(T_int)
            results['T_mass'].append(T_mass)
            results['T_exterior'].append(T_ext)
            results['heating_demand'].append(heating)
            results['cooling_demand'].append(cooling)
            results['thermal_lag'].append(lag)
            
            # Update initial conditions for next hour
            y0 = y_final
        
        # Calculate summary statistics
        results['summary'] = {
            'avg_interior_temp': np.mean(results['T_interior']),
            'temp_fluctuation': np.std(results['T_interior']),
            'total_heating_kwh': sum(results['heating_demand']) / 1000,
            'total_cooling_kwh': sum(results['cooling_demand']) / 1000,
            'thermal_autonomy_hours': sum(1 for h, c in zip(results['heating_demand'], 
                                                           results['cooling_demand']) 
                                         if h == 0 and c == 0),
            'avg_thermal_lag': np.mean(np.abs(results['thermal_lag']))
        }
        
        return results
    
    def calculate_time_constant(self) -> float:
        """Calculate the thermal time constant of the building"""
        # Effective resistance seen by thermal mass
        R_eff = 1.0 / (1/self.R_ms + 1/self.R_em)
        
        # Time constant τ = R × C
        tau = R_eff * self.C_m
        
        # Convert to hours
        return tau / 3600
    
    def get_thermal_properties(self) -> Dict:
        """Get summary of thermal properties"""
        return {
            'thermal_mass_kJ/K': self.C_m / 1000,
            'time_constant_hours': self.calculate_time_constant(),
            'total_thermal_resistance_K/W': self.R_si + self.R_ms,
            'window_heat_loss_coefficient_W/K': 1.0 / self.R_window,
            'ventilation_heat_loss_coefficient_W/K': 1.0 / self.R_ventilation,
            'effective_U_value_W/m2K': 1.0 / ((self.R_si + self.R_ms) * 
                                             (self.params['wall_area'] + self.params['window_area']))
        }

    def export_boundary_conditions(self, results: Dict = None) -> Dict:
        """
        Export boundary conditions for CHT/LBM simulation.
        Returns the temperature of the inner wall surface (Tsi).
        
        Args:
            results: Optional results from simulate_period. 
                     If None, returns current state of self.nodes['mass']
        
        Returns:
            Dict with 'wall_temperature', 'window_temperature', etc.
        """
        if results:
            # Return the last calculated value from simulation
            t_mass = results['T_mass'][-1]
            t_int = results['T_interior'][-1]
            t_ext = results['T_exterior'][-1]
        else:
            t_mass = self.nodes['mass'].temperature
            t_int = self.nodes['interior'].temperature
            t_ext = self.nodes['exterior'].temperature
            
        # In this 5R1C topology, T_mass is the node representing the massive wall.
        # We use it as the boundary temperature for the Fluid (LBM).
        
        # Calculate Window inner surface temperature (steady state approx for window)
        # Q_win = U * A * (Tint - Text)
        # T_win_inner = Tint - (Q_win / A) * R_si_win
        # R_si_win is roughly 0.13
        u_win = self.params.get('u_value_window', 2.0)
        # T_win_surf = T_int - (T_int - T_ext) * (U_win * 0.13)
        t_window = t_int - (t_int - t_ext) * (u_win * 0.13)
        
        return {
            'wall_temperature': t_mass,
            'window_temperature': t_window,
            'air_temperature': t_int,
            'timestamp': 'latest'
        }



# Example usage function
def demo_rc_network():
    """Demonstration of RC-Network thermal simulation"""
    
    # Building parameters for a typical room
    building_params = {
        'wall_area': 60,  # m²
        'window_area': 6,  # m²
        'floor_area': 25,  # m²
        'volume': 75,  # m³
        'wall_thickness': 0.3,  # m
        'wall_material': {
            'density': 2400,  # kg/m³ (concrete)
            'specific_heat': 1000,  # J/kg·K
            'conductivity': 1.7  # W/m·K
        },
        'u_value_wall': 0.3,  # W/m²K (insulated wall)
        'u_value_window': 1.2,  # W/m²K (triple glazing)
        'air_change_rate': 0.5  # 1/h
    }
    
    # Create model
    model = RCNetworkModel(building_params)
    
    # Create sample weather data (24 hours)
    hours = 24
    weather_data = {
        'temperature': [10 + 5*np.sin(2*np.pi*h/24 - np.pi/2) for h in range(hours)],
        'solar_radiation': [max(0, 500*np.sin(2*np.pi*h/24 - np.pi/2)) for h in range(hours)]
    }
    
    # Run simulation
    results = model.simulate_period(weather_data)
    
    # Print results
    print("RC-Network Thermal Simulation Results")
    print("=" * 50)
    print(f"Average interior temperature: {results['summary']['avg_interior_temp']:.1f}°C")
    print(f"Temperature fluctuation (std): {results['summary']['temp_fluctuation']:.2f}°C")
    print(f"Total heating demand: {results['summary']['total_heating_kwh']:.1f} kWh")
    print(f"Thermal autonomy: {results['summary']['thermal_autonomy_hours']}/{hours} hours")
    print(f"Thermal time constant: {model.calculate_time_constant():.1f} hours")
    
    return results


if __name__ == "__main__":
    demo_rc_network()

