"""
EPW (EnergyPlus Weather) File Parser
Extracts hourly weather data for thermal simulations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os


class EPWParser:
    """
    Parser for EnergyPlus Weather (EPW) files
    
    EPW files contain hourly weather data including:
    - Temperature
    - Humidity
    - Solar radiation (direct and diffuse)
    - Wind speed and direction
    - Atmospheric pressure
    """
    
    # EPW column definitions (after header)
    EPW_COLUMNS = {
        'year': 0,
        'month': 1, 
        'day': 2,
        'hour': 3,
        'dry_bulb_temp': 6,  # °C
        'dew_point_temp': 7,  # °C
        'relative_humidity': 8,  # %
        'atmospheric_pressure': 9,  # Pa
        'horizontal_infrared': 12,  # Wh/m²
        'direct_normal_radiation': 14,  # Wh/m²
        'diffuse_horizontal_radiation': 15,  # Wh/m²
        'global_horizontal_radiation': 13,  # Wh/m²
        'wind_direction': 20,  # degrees
        'wind_speed': 21,  # m/s
        'sky_cover': 22,  # tenths
        'liquid_precipitation': 33  # mm
    }
    
    def __init__(self, epw_file_path: Optional[str] = None):
        """
        Initialize EPW Parser
        
        Args:
            epw_file_path: Path to EPW file (optional)
        """
        self.file_path = epw_file_path
        self.header = {}
        self.data = None
        self.location = {}
        
        if epw_file_path and os.path.exists(epw_file_path):
            self.parse_file()
    
    def parse_file(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Parse EPW file and extract weather data
        
        Args:
            file_path: Path to EPW file (uses stored path if not provided)
        
        Returns:
            DataFrame with parsed weather data
        """
        if file_path:
            self.file_path = file_path
        
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"EPW file not found: {self.file_path}")
        
        # Read file
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header (first 8 lines)
        self._parse_header(lines[:8])
        
        # Parse weather data
        weather_lines = lines[8:]
        data_rows = []
        
        for line in weather_lines:
            values = line.strip().split(',')
            if len(values) > 35:  # Valid data row
                data_rows.append(values)
        
        # Create DataFrame
        self.data = pd.DataFrame(data_rows)
        
        # Extract relevant columns
        extracted_data = {
            'datetime': self._create_datetime_index(self.data),
            'temperature': self.data.iloc[:, self.EPW_COLUMNS['dry_bulb_temp']].astype(float),
            'dew_point': self.data.iloc[:, self.EPW_COLUMNS['dew_point_temp']].astype(float),
            'humidity': self.data.iloc[:, self.EPW_COLUMNS['relative_humidity']].astype(float),
            'pressure': self.data.iloc[:, self.EPW_COLUMNS['atmospheric_pressure']].astype(float),
            'direct_radiation': self.data.iloc[:, self.EPW_COLUMNS['direct_normal_radiation']].astype(float),
            'diffuse_radiation': self.data.iloc[:, self.EPW_COLUMNS['diffuse_horizontal_radiation']].astype(float),
            'global_radiation': self.data.iloc[:, self.EPW_COLUMNS['global_horizontal_radiation']].astype(float),
            'wind_speed': self.data.iloc[:, self.EPW_COLUMNS['wind_speed']].astype(float),
            'wind_direction': self.data.iloc[:, self.EPW_COLUMNS['wind_direction']].astype(float)
        }
        
        self.data = pd.DataFrame(extracted_data)
        self.data.set_index('datetime', inplace=True)
        
        return self.data
    
    def _parse_header(self, header_lines: List[str]):
        """Parse EPW header information"""
        
        # Line 1: Location
        location_parts = header_lines[0].strip().split(',')
        self.location = {
            'city': location_parts[1] if len(location_parts) > 1 else '',
            'state': location_parts[2] if len(location_parts) > 2 else '',
            'country': location_parts[3] if len(location_parts) > 3 else '',
            'data_source': location_parts[4] if len(location_parts) > 4 else '',
            'wmo_number': location_parts[5] if len(location_parts) > 5 else '',
            'latitude': float(location_parts[6]) if len(location_parts) > 6 else 0,
            'longitude': float(location_parts[7]) if len(location_parts) > 7 else 0,
            'timezone': float(location_parts[8]) if len(location_parts) > 8 else 0,
            'elevation': float(location_parts[9]) if len(location_parts) > 9 else 0
        }
        
        self.header['location'] = self.location
    
    def _create_datetime_index(self, data: pd.DataFrame) -> List[datetime]:
        """Create datetime index from EPW data columns"""
        datetimes = []
        
        for idx, row in data.iterrows():
            try:
                year = int(row.iloc[self.EPW_COLUMNS['year']])
                # Use a fixed year if data year is invalid
                if year < 1900:
                    year = 2023
                    
                month = int(row.iloc[self.EPW_COLUMNS['month']])
                day = int(row.iloc[self.EPW_COLUMNS['day']])
                hour = int(row.iloc[self.EPW_COLUMNS['hour']]) - 1  # EPW uses 1-24
                
                dt = datetime(year, month, day, hour)
                datetimes.append(dt)
            except:
                # If parsing fails, use a sequential datetime
                if datetimes:
                    dt = datetimes[-1] + timedelta(hours=1)
                else:
                    dt = datetime(2023, 1, 1, 0)
                datetimes.append(dt)
        
        return datetimes
    
    def get_design_days(self) -> Dict:
        """
        Extract design day conditions (hottest and coldest days)
        
        Returns:
            Dictionary with design day conditions
        """
        if self.data is None:
            raise ValueError("No data loaded. Parse EPW file first.")
        
        # Find extreme temperature days
        daily_avg = self.data.groupby(self.data.index.date)['temperature'].mean()
        
        hottest_day = daily_avg.idxmax()
        coldest_day = daily_avg.idxmin()
        
        # Extract 24-hour data for design days
        hot_day_data = self.data[self.data.index.date == hottest_day]
        cold_day_data = self.data[self.data.index.date == coldest_day]
        
        return {
            'summer_design_day': {
                'date': str(hottest_day),
                'max_temp': hot_day_data['temperature'].max(),
                'min_temp': hot_day_data['temperature'].min(),
                'avg_temp': hot_day_data['temperature'].mean(),
                'solar_radiation': hot_day_data['global_radiation'].sum(),
                'hourly_temps': hot_day_data['temperature'].tolist(),
                'hourly_radiation': hot_day_data['global_radiation'].tolist()
            },
            'winter_design_day': {
                'date': str(coldest_day),
                'max_temp': cold_day_data['temperature'].max(),
                'min_temp': cold_day_data['temperature'].min(),
                'avg_temp': cold_day_data['temperature'].mean(),
                'solar_radiation': cold_day_data['global_radiation'].sum(),
                'hourly_temps': cold_day_data['temperature'].tolist(),
                'hourly_radiation': cold_day_data['global_radiation'].tolist()
            }
        }
    
    def get_monthly_statistics(self) -> pd.DataFrame:
        """
        Calculate monthly weather statistics
        
        Returns:
            DataFrame with monthly statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Parse EPW file first.")
        
        monthly_stats = self.data.groupby(self.data.index.month).agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'humidity': 'mean',
            'global_radiation': 'sum',
            'wind_speed': 'mean'
        })
        
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
        monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return monthly_stats
    
    def get_typical_week(self, month: int) -> Dict:
        """
        Extract a typical week for a given month
        
        Args:
            month: Month number (1-12)
        
        Returns:
            Dictionary with typical week data
        """
        if self.data is None:
            raise ValueError("No data loaded. Parse EPW file first.")
        
        # Filter data for specified month
        month_data = self.data[self.data.index.month == month]
        
        # Find week with temperature closest to monthly average
        monthly_avg_temp = month_data['temperature'].mean()
        
        # Group by week
        weekly_avg = month_data.groupby(month_data.index.isocalendar().week)['temperature'].mean()
        typical_week_num = (weekly_avg - monthly_avg_temp).abs().idxmin()
        
        # Extract typical week data
        typical_week_data = month_data[month_data.index.isocalendar().week == typical_week_num]
        
        # Ensure we have exactly 7 days (168 hours)
        if len(typical_week_data) > 168:
            typical_week_data = typical_week_data.iloc[:168]
        
        return {
            'month': month,
            'week_number': typical_week_num,
            'temperature': typical_week_data['temperature'].tolist(),
            'humidity': typical_week_data['humidity'].tolist(),
            'solar_radiation': typical_week_data['global_radiation'].tolist(),
            'wind_speed': typical_week_data['wind_speed'].tolist()
        }
    
    def get_heating_cooling_degree_days(self, base_temp: float = 18.0) -> Dict:
        """
        Calculate heating and cooling degree days
        
        Args:
            base_temp: Base temperature for calculation (default 18°C)
        
        Returns:
            Dictionary with degree days
        """
        if self.data is None:
            raise ValueError("No data loaded. Parse EPW file first.")
        
        daily_avg = self.data.groupby(self.data.index.date)['temperature'].mean()
        
        # Heating degree days (HDD)
        hdd = (base_temp - daily_avg).clip(lower=0).sum()
        
        # Cooling degree days (CDD)
        cdd = (daily_avg - base_temp).clip(lower=0).sum()
        
        # Monthly breakdown
        monthly_hdd = {}
        monthly_cdd = {}
        
        for month in range(1, 13):
            month_data = self.data[self.data.index.month == month]
            month_daily_avg = month_data.groupby(month_data.index.date)['temperature'].mean()
            
            monthly_hdd[month] = (base_temp - month_daily_avg).clip(lower=0).sum()
            monthly_cdd[month] = (month_daily_avg - base_temp).clip(lower=0).sum()
        
        return {
            'annual_hdd': hdd,
            'annual_cdd': cdd,
            'monthly_hdd': monthly_hdd,
            'monthly_cdd': monthly_cdd,
            'base_temperature': base_temp
        }
    
    def export_for_rc_model(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Dict:
        """
        Export weather data in format suitable for RC-Network model
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary with weather data for RC model
        """
        if self.data is None:
            raise ValueError("No data loaded. Parse EPW file first.")
        
        # Filter date range if specified
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        return {
            'temperature': data['temperature'].tolist(),
            'solar_radiation': data['global_radiation'].tolist(),
            'wind_speed': data['wind_speed'].tolist(),
            'humidity': data['humidity'].tolist(),
            'timestamps': [str(dt) for dt in data.index]
        }


class SyntheticWeatherGenerator:
    """
    Generate synthetic weather data when EPW files are not available
    Based on location and typical climate patterns
    """
    
    @staticmethod
    def generate_typical_year(latitude: float, climate_type: str = 'temperate') -> Dict:
        """
        Generate synthetic typical meteorological year data
        
        Args:
            latitude: Location latitude
            climate_type: 'temperate', 'hot', 'cold', 'tropical'
        
        Returns:
            Dictionary with synthetic weather data
        """
        hours_per_year = 8760
        
        # Base temperature patterns by climate
        climate_params = {
            'temperate': {'avg': 15, 'amplitude': 10, 'daily_swing': 8},
            'hot': {'avg': 25, 'amplitude': 8, 'daily_swing': 10},
            'cold': {'avg': 5, 'amplitude': 15, 'daily_swing': 6},
            'tropical': {'avg': 27, 'amplitude': 3, 'daily_swing': 5}
        }
        
        params = climate_params.get(climate_type, climate_params['temperate'])
        
        # Generate hourly data
        temperatures = []
        solar_radiation = []
        
        for hour in range(hours_per_year):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            
            # Annual temperature variation (cosine)
            annual_factor = np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak at summer solstice
            
            # Daily temperature variation (sine)
            daily_factor = np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Min at 6am, max at 6pm
            
            # Calculate temperature
            temp = (params['avg'] + 
                   params['amplitude'] * annual_factor +
                   params['daily_swing'] * daily_factor * 0.5 +
                   np.random.normal(0, 1))  # Random variation
            
            temperatures.append(temp)
            
            # Solar radiation (simplified)
            if 6 <= hour_of_day <= 18:  # Daylight hours
                solar_angle = np.sin(np.pi * (hour_of_day - 6) / 12)
                base_radiation = 800 * solar_angle  # Max 800 W/m²
                
                # Seasonal adjustment
                seasonal_factor = 1 + 0.3 * annual_factor
                
                # Cloud cover (random)
                cloud_factor = np.random.uniform(0.3, 1.0)
                
                radiation = base_radiation * seasonal_factor * cloud_factor
            else:
                radiation = 0
            
            solar_radiation.append(max(0, radiation))
        
        return {
            'temperature': temperatures,
            'solar_radiation': solar_radiation,
            'location': {
                'latitude': latitude,
                'climate_type': climate_type
            }
        }


# Example usage
def demo_weather_parser():
    """Demonstration of weather data parsing"""
    
    # Generate synthetic weather data
    print("Generating synthetic weather data for Moscow (55.75°N)...")
    generator = SyntheticWeatherGenerator()
    synthetic_data = generator.generate_typical_year(55.75, 'cold')
    
    # Extract sample period (first week of January)
    sample_temps = synthetic_data['temperature'][:168]  # First week
    sample_solar = synthetic_data['solar_radiation'][:168]
    
    print(f"\nFirst week of January:")
    print(f"Average temperature: {np.mean(sample_temps):.1f}°C")
    print(f"Min temperature: {np.min(sample_temps):.1f}°C")
    print(f"Max temperature: {np.max(sample_temps):.1f}°C")
    print(f"Total solar radiation: {np.sum(sample_solar):.0f} Wh/m²")
    
    return synthetic_data


if __name__ == "__main__":
    demo_weather_parser()
