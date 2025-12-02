"""
AHI 2.0 Ultimate - Synthetic EPW Weather Generator

Генератор синтетических погодных файлов EPW для сценарного моделирования.
Позволяет создавать сценарии 'What-if' для тестирования архитектурной гармонии.

Основано на:
- ASHRAE Clear Sky Model для солнечной радиации
- Magnus formula для психрометрии
- Erbs et al. correlation для разделения DNI/DHI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class Location:
    """Геолокация для расчетов солнечной позиции"""
    latitude: float      # φ в градусах (-90 to 90)
    longitude: float     # λ в градусах (-180 to 180)
    timezone: int        # Часовой пояс UTC offset
    elevation: float     # Высота над уровнем моря (м)
    name: str = "Custom Location"


class SolarPositionCalculator:
    """
    Упрощенный расчет позиции солнца (NOAA algorithm)
    Для полной точности использовать SPA (Solar Position Algorithm)
    """
    
    @staticmethod
    def day_of_year(month: int, day: int, year: int) -> int:
        """Номер дня в году"""
        return (datetime(year, month, day) - datetime(year, 1, 1)).days + 1
    
    @staticmethod
    def solar_declination(day_of_year: int) -> float:
        """Склонение солнца δ в радианах"""
        # Приближение Cooper equation
        return math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    
    @staticmethod
    def equation_of_time(day_of_year: int) -> float:
        """Уравнение времени в минутах"""
        B = math.radians(360 * (day_of_year - 81) / 365)
        return 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    
    @staticmethod
    def hour_angle(hour: float, longitude: float, timezone: int, eot: float) -> float:
        """Часовой угол ω в радианах"""
        # Solar time
        lstm = 15 * timezone  # Local Standard Time Meridian
        time_correction = 4 * (longitude - lstm) + eot  # в минутах
        lst = hour + time_correction / 60  # Local Solar Time
        return math.radians(15 * (lst - 12))
    
    def calculate_zenith_angle(self, lat: float, day_of_year: int, hour: float, 
                                longitude: float, timezone: int) -> float:
        """
        Расчет зенитного угла θ_z
        
        Returns:
            Зенитный угол в градусах (0 = солнце в зените, 90 = горизонт)
        """
        lat_rad = math.radians(lat)
        decl = self.solar_declination(day_of_year)
        eot = self.equation_of_time(day_of_year)
        omega = self.hour_angle(hour, longitude, timezone, eot)
        
        # cos(θ_z) = sin(φ)*sin(δ) + cos(φ)*cos(δ)*cos(ω)
        cos_zenith = (math.sin(lat_rad) * math.sin(decl) + 
                      math.cos(lat_rad) * math.cos(decl) * math.cos(omega))
        
        # Ограничиваем для численной стабильности
        cos_zenith = max(-1, min(1, cos_zenith))
        
        return math.degrees(math.acos(cos_zenith))


class ASHRAEClearSky:
    """
    ASHRAE Clear Sky Model для расчета солнечной радиации
    """
    
    def __init__(self, location: Location):
        self.location = location
        self.solar_calc = SolarPositionCalculator()
    
    def extraterrestrial_radiation(self, day_of_year: int) -> float:
        """
        Экстерриториальная радиация I_0 (Вт/м²)
        Учитывает эллиптичность орбиты Земли
        """
        I_sc = 1367  # Солнечная постоянная (Вт/м²)
        B = 2 * math.pi * (day_of_year - 1) / 365
        
        # Коррекция на расстояние до Солнца
        E_0 = 1.00011 + 0.034221 * math.cos(B) + 0.00128 * math.sin(B) + \
              0.000719 * math.cos(2*B) + 0.000077 * math.sin(2*B)
        
        return I_sc * E_0
    
    def optical_air_mass(self, zenith_angle: float) -> float:
        """
        Оптическая масса воздуха (Kasten & Young, 1989)
        """
        if zenith_angle >= 90:
            return float('inf')
        
        z_rad = math.radians(zenith_angle)
        # Более точная формула с учетом рефракции
        return 1 / (math.cos(z_rad) + 0.50572 * pow(96.07995 - zenith_angle, -1.6364))
    
    def clear_sky_dni(self, day_of_year: int, hour: float) -> float:
        """
        Direct Normal Irradiance для чистого неба (Вт/м²)
        ASHRAE Clear Sky Model
        """
        zenith = self.solar_calc.calculate_zenith_angle(
            self.location.latitude, day_of_year, hour,
            self.location.longitude, self.location.timezone
        )
        
        if zenith >= 90:
            return 0.0
        
        I_0 = self.extraterrestrial_radiation(day_of_year)
        m = self.optical_air_mass(zenith)
        
        # Optical depths (типичные значения для чистого неба)
        tau_b = 0.32  # Прямая (зависит от влажности и аэрозолей)
        ab = 1.454 - 0.406 * tau_b - 0.268 * tau_b**2 + 0.021 * tau_b**3
        
        # DNI = I_0 * exp(-tau_b * m^ab)
        DNI = I_0 * math.exp(-tau_b * pow(m, ab))
        
        return max(0, DNI)
    
    def clear_sky_dhi(self, day_of_year: int, hour: float) -> float:
        """
        Diffuse Horizontal Irradiance для чистого неба (Вт/м²)
        """
        zenith = self.solar_calc.calculate_zenith_angle(
            self.location.latitude, day_of_year, hour,
            self.location.longitude, self.location.timezone
        )
        
        if zenith >= 90:
            return 0.0
        
        I_0 = self.extraterrestrial_radiation(day_of_year)
        m = self.optical_air_mass(zenith)
        
        # Diffuse optical depth
        tau_d = 0.4  # Диффузный
        ad = 0.507 + 0.205 * tau_d - 0.080 * tau_d**2 - 0.190 * tau_d**3
        
        # DHI = I_0 * exp(-tau_d * m^ad)
        DHI = I_0 * math.exp(-tau_d * pow(m, ad)) * math.cos(math.radians(zenith))
        
        return max(0, DHI)
    
    def calculate_ghi(self, day_of_year: int, hour: float, cloud_cover: float = 0.0) -> Tuple[float, float, float]:
        """
        Global Horizontal Irradiance с учетом облачности
        
        Args:
            day_of_year: День года (1-365)
            hour: Час дня (0-23)
            cloud_cover: Облачность (0.0 - 1.0)
            
        Returns:
            Tuple[GHI, DNI, DHI] в Вт/м²
        """
        zenith = self.solar_calc.calculate_zenith_angle(
            self.location.latitude, day_of_year, hour,
            self.location.longitude, self.location.timezone
        )
        
        if zenith >= 90:
            return 0.0, 0.0, 0.0
        
        # Чистое небо
        DNI_clear = self.clear_sky_dni(day_of_year, hour)
        DHI_clear = self.clear_sky_dhi(day_of_year, hour)
        
        # GHI_clear = DNI * cos(θ_z) + DHI
        GHI_clear = DNI_clear * math.cos(math.radians(zenith)) + DHI_clear
        
        # Влияние облачности: GHI = GHI_clear * (1 - 0.75 * CC^3.4)
        # Эмпирическая формула Kasten & Czeplak
        cloud_factor = 1 - 0.75 * pow(cloud_cover, 3.4)
        
        GHI = GHI_clear * cloud_factor
        
        # При облачности меняется соотношение DNI/DHI
        # Используем модель Erbs et al. для разделения
        if GHI > 0 and GHI_clear > 0:
            k_t = GHI / self.extraterrestrial_radiation(day_of_year)
            k_t = max(0, min(1, k_t))
            
            # Erbs correlation для diffuse fraction
            if k_t <= 0.22:
                diffuse_fraction = 1.0 - 0.09 * k_t
            elif k_t <= 0.80:
                diffuse_fraction = 0.9511 - 0.1604*k_t + 4.388*k_t**2 - 16.638*k_t**3 + 12.336*k_t**4
            else:
                diffuse_fraction = 0.165
            
            DHI = GHI * diffuse_fraction
            DNI = (GHI - DHI) / max(0.01, math.cos(math.radians(zenith)))
        else:
            DNI = 0
            DHI = 0
        
        return GHI, max(0, DNI), max(0, DHI)


class SyntheticEPWGenerator:
    """
    Генератор синтетических погодных файлов EPW для AHI 2.0.
    Позволяет создавать сценарии 'What-if' для тестирования архитектурной гармонии.
    """
    
    def __init__(self, lat: float, lon: float, year: int = 2024, 
                 timezone: int = 0, elevation: float = 0, name: str = "Custom"):
        """
        Args:
            lat: Широта (-90 to 90)
            lon: Долгота (-180 to 180)
            year: Год для генерации
            timezone: Часовой пояс (UTC offset)
            elevation: Высота над уровнем моря (м)
            name: Название локации
        """
        self.location = Location(lat, lon, timezone, elevation, name)
        self.year = year
        self.sky_model = ASHRAEClearSky(self.location)
    
    def generate_temperature_profile(self, t_min: float, t_max: float, 
                                      hours: np.ndarray) -> np.ndarray:
        """
        Генерация суточного профиля температуры.
        
        Синусоидальная модель с фазовым сдвигом:
        T(t) = T_mean + T_amp * cos(π(t - t_lag)/12)
        
        Пик температуры в ~15:00, минимум в ~03:00
        
        Args:
            t_min: Минимальная температура (°C)
            t_max: Максимальная температура (°C)
            hours: Массив часов (0-23)
            
        Returns:
            Массив температур (°C)
        """
        t_amp = (t_max - t_min) / 2
        t_mean = (t_max + t_min) / 2
        t_lag = 15  # Час пика температуры (фазовый сдвиг из-за тепловой инерции)
        
        # T(t) = T_mean + T_amp * cos(π(t - t_lag)/12)
        dry_bulb_temp = t_mean + t_amp * np.cos((hours - t_lag) * np.pi / 12)
        
        return dry_bulb_temp
    
    def calculate_humidity(self, dry_bulb_temp: np.ndarray, 
                           rh_avg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет влажности и точки росы.
        
        Использует формулу Магнуса для давления насыщенного пара:
        e_s = 6.112 * exp(17.67 * T / (T + 243.5))
        
        Точка росы:
        T_dew = (243.5 * ln(e/6.112)) / (17.67 - ln(e/6.112))
        
        Args:
            dry_bulb_temp: Температура воздуха (°C)
            rh_avg: Средняя относительная влажность (0-100)
            
        Returns:
            Tuple[relative_humidity, dew_point]
        """
        # При постоянной абсолютной влажности RH обратно пропорционален T
        # Упрощение: используем среднюю RH
        rh = np.full_like(dry_bulb_temp, rh_avg)
        
        # Давление насыщенного пара (формула Магнуса)
        es = 6.112 * np.exp((17.67 * dry_bulb_temp) / (dry_bulb_temp + 243.5))
        
        # Парциальное давление пара
        e = es * (rh / 100.0)
        
        # Точка росы (обратная формула Магнуса)
        # Защита от log(0)
        e = np.maximum(e, 0.001)
        dew_point = (243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))
        
        return rh, dew_point
    
    def generate_radiation(self, month: int, day: int, hours: np.ndarray,
                           cloud_cover: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерация солнечной радиации.
        
        Args:
            month: Месяц (1-12)
            day: День месяца
            hours: Массив часов
            cloud_cover: Облачность (0.0 - 1.0)
            
        Returns:
            Tuple[GHI, DNI, DHI] arrays в Вт/м²
        """
        doy = self.sky_model.solar_calc.day_of_year(month, day, self.year)
        
        ghi = np.zeros_like(hours, dtype=float)
        dni = np.zeros_like(hours, dtype=float)
        dhi = np.zeros_like(hours, dtype=float)
        
        for i, hour in enumerate(hours):
            g, d_n, d_h = self.sky_model.calculate_ghi(doy, float(hour), cloud_cover)
            ghi[i] = g
            dni[i] = d_n
            dhi[i] = d_h
        
        return ghi, dni, dhi
    
    def generate_day_profile(self, month: int, day: int, 
                              t_min: float, t_max: float,
                              cloud_cover: float, rh_avg: float) -> pd.DataFrame:
        """
        Генерация полного суточного профиля погоды.
        
        Args:
            month: Месяц (1-12)
            day: День месяца
            t_min: Минимальная температура (°C)
            t_max: Максимальная температура (°C)
            cloud_cover: Облачность (0.0 - 1.0)
            rh_avg: Средняя относительная влажность (%)
            
        Returns:
            DataFrame с почасовыми данными
        """
        # 1. Создание временной шкалы (24 часа)
        hours = np.arange(24)
        
        # 2. Генерация температурной кривой (Синусоида с лагом)
        dry_bulb_temp = self.generate_temperature_profile(t_min, t_max, hours)
        
        # 3. Генерация влажности и точки росы
        rel_humidity, dew_point = self.calculate_humidity(dry_bulb_temp, rh_avg)
        
        # 4. Генерация радиации (ASHRAE Clear Sky + облачность)
        ghi, dni, dhi = self.generate_radiation(month, day, hours, cloud_cover)
        
        # 5. Формирование DataFrame в структуре EPW
        epw_data = pd.DataFrame({
            'Year': [self.year] * 24,
            'Month': [month] * 24,
            'Day': [day] * 24,
            'Hour': hours + 1,  # EPW использует 1-24
            'Minute': [0] * 24,
            
            # Температуры
            'DryBulb': np.round(dry_bulb_temp, 1),
            'DewPoint': np.round(dew_point, 1),
            'RelHum': np.round(rel_humidity, 0).astype(int),
            
            # Давление (стандартное с коррекцией высоты)
            'AtmosPressure': [int(101325 * np.exp(-self.location.elevation / 8500))] * 24,
            
            # Радиация (экстерриториальная - для валидации)
            'ExtHorzRad': [0] * 24,  # Заполняется отдельно
            'ExtDirRad': [0] * 24,
            'HorzIRSky': [0] * 24,
            
            # Солнечная радиация
            'GloHorzRad': np.round(ghi, 0).astype(int),
            'DirNormRad': np.round(dni, 0).astype(int),
            'DifHorzRad': np.round(dhi, 0).astype(int),
            
            # Освещенность (приблизительно из радиации)
            'GloHorzIllum': (ghi * 110).astype(int),  # ~110 lm/W для дневного света
            'DirNormIllum': (dni * 110).astype(int),
            'DifHorzIllum': (dhi * 110).astype(int),
            'ZenLum': [0] * 24,
            
            # Ветер (по умолчанию)
            'WindDir': [180] * 24,
            'WindSpd': [2.0] * 24,
            
            # Облачность
            'TotSkyCvr': [int(cloud_cover * 10)] * 24,
            'OpaqSkyCvr': [int(cloud_cover * 10)] * 24,
            
            # Видимость, высота облаков и т.д.
            'Visibility': [9999] * 24,
            'CeilHgt': [77777] * 24,
            'PresWeathObs': [0] * 24,
            'PresWeathCodes': [''] * 24,
            'PrecipWtr': [0] * 24,
            'AerosolOptDepth': [0.1] * 24,
            'SnowDepth': [0] * 24,
            'DaysLastSnow': [99] * 24,
            'Albedo': [0.2] * 24,
            'Rain': [0] * 24,
            'RainQty': [0] * 24,
        })
        
        return epw_data
    
    def generate_scenario(self, scenario_name: str, 
                          start_month: int, start_day: int,
                          num_days: int,
                          t_min_base: float, t_max_base: float,
                          cloud_cover: float = 0.3,
                          rh_avg: float = 50.0,
                          heat_wave_day: Optional[int] = None,
                          heat_wave_delta: float = 10.0) -> pd.DataFrame:
        """
        Генерация сценария погоды для нескольких дней.
        
        Поддержка сценариев:
        - "Тепловая волна 2050 года"
        - "Внезапный зимний шторм"
        - Обычные условия
        
        Args:
            scenario_name: Название сценария
            start_month, start_day: Начальная дата
            num_days: Количество дней
            t_min_base, t_max_base: Базовые температуры
            cloud_cover: Облачность (0.0-1.0)
            rh_avg: Влажность (%)
            heat_wave_day: День с тепловой волной (None = нет)
            heat_wave_delta: Прирост температуры при тепловой волне
            
        Returns:
            DataFrame со всеми днями
        """
        all_days = []
        
        current_date = datetime(self.year, start_month, start_day)
        
        for day_num in range(num_days):
            # Корректируем температуры для сценариев
            t_min = t_min_base
            t_max = t_max_base
            cc = cloud_cover
            
            # Сценарий: тепловая волна
            if heat_wave_day is not None and day_num == heat_wave_day:
                t_min += heat_wave_delta * 0.6
                t_max += heat_wave_delta
                cc = min(cc, 0.2)  # Меньше облаков в жару
            
            # Генерируем день
            day_data = self.generate_day_profile(
                current_date.month, current_date.day,
                t_min, t_max, cc, rh_avg
            )
            
            all_days.append(day_data)
            current_date += timedelta(days=1)
        
        result = pd.concat(all_days, ignore_index=True)
        result.attrs['scenario_name'] = scenario_name
        result.attrs['location'] = self.location.name
        
        return result
    
    def save_to_epw(self, data: pd.DataFrame, filename: str = "temp_scenario.epw"):
        """
        Сохранение в формате EPW (EnergyPlus Weather).
        
        Args:
            data: DataFrame с погодными данными
            filename: Имя файла
        """
        # EPW Header (8 строк)
        header = [
            f"LOCATION,{self.location.name},Custom,Custom,Custom,{self.location.timezone},"
            f"{self.location.latitude},{self.location.longitude},{self.location.timezone},"
            f"{self.location.elevation}",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Synthetic EPW generated by AHI 2.0",
            "COMMENTS 2,For scenario modeling only",
            "DATA PERIODS,1,1,Data,Sunday,1/1,12/31"
        ]
        
        with open(filename, 'w', newline='') as f:
            # Записываем заголовок
            for line in header:
                f.write(line + '\n')
            
            # Записываем данные
            # EPW формат: все колонки через запятую
            epw_columns = [
                'Year', 'Month', 'Day', 'Hour', 'Minute',
                'DryBulb', 'DewPoint', 'RelHum', 'AtmosPressure',
                'ExtHorzRad', 'ExtDirRad', 'HorzIRSky',
                'GloHorzRad', 'DirNormRad', 'DifHorzRad',
                'GloHorzIllum', 'DirNormIllum', 'DifHorzIllum', 'ZenLum',
                'WindDir', 'WindSpd', 'TotSkyCvr', 'OpaqSkyCvr',
                'Visibility', 'CeilHgt', 'PresWeathObs', 'PresWeathCodes',
                'PrecipWtr', 'AerosolOptDepth', 'SnowDepth', 'DaysLastSnow',
                'Albedo', 'Rain', 'RainQty'
            ]
            
            for _, row in data.iterrows():
                values = [str(row[col]) if col in row else '0' for col in epw_columns]
                f.write(','.join(values) + '\n')
        
        print(f"[SyntheticEPWGenerator] Saved EPW file: {filename}")
        return filename
    
    def get_boundary_conditions(self, month: int, day: int, hour: int,
                                 t_min: float, t_max: float,
                                 cloud_cover: float, rh_avg: float) -> Dict:
        """
        Получить граничные условия для CHT solver.
        
        Возвращает данные в формате, совместимом с ThermalBoundaryConditions.
        
        Args:
            month, day, hour: Время
            t_min, t_max: Температурный диапазон (°C)
            cloud_cover: Облачность (0.0-1.0)
            rh_avg: Влажность (%)
            
        Returns:
            Dict с граничными условиями
        """
        # Генерируем профиль на час
        hours = np.array([hour])
        temp = self.generate_temperature_profile(t_min, t_max, hours)[0]
        rh, dew = self.calculate_humidity(np.array([temp]), rh_avg)
        
        doy = self.sky_model.solar_calc.day_of_year(month, day, self.year)
        ghi, dni, dhi = self.sky_model.calculate_ghi(doy, float(hour), cloud_cover)
        
        return {
            'external_temperature': temp + 273.15,  # К
            'relative_humidity': rh[0] / 100.0,     # 0-1
            'dew_point': dew[0] + 273.15,           # К
            'ghi': ghi,                              # Вт/м²
            'dni': dni,
            'dhi': dhi,
            'cloud_cover': cloud_cover,
            'timestamp': datetime(self.year, month, day, hour).isoformat()
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Создаем генератор для Токио
    generator = SyntheticEPWGenerator(
        lat=35.6762,
        lon=139.6503,
        timezone=9,
        elevation=40,
        name="Tokyo"
    )
    
    # Сценарий 1: Обычный летний день
    print("\n=== Сценарий: Обычный летний день в Токио ===")
    normal_day = generator.generate_day_profile(
        month=7, day=15,
        t_min=25.0, t_max=33.0,
        cloud_cover=0.3,
        rh_avg=70.0
    )
    print(normal_day[['Hour', 'DryBulb', 'RelHum', 'GloHorzRad', 'DirNormRad']].head(12))
    
    # Сценарий 2: Тепловая волна 2050
    print("\n=== Сценарий: Тепловая волна 2050 ===")
    heat_wave = generator.generate_scenario(
        scenario_name="Heat Wave 2050",
        start_month=7, start_day=20,
        num_days=5,
        t_min_base=28.0, t_max_base=35.0,
        cloud_cover=0.1,
        rh_avg=75.0,
        heat_wave_day=2,  # 3-й день - пик волны
        heat_wave_delta=8.0  # +8°C к максимуму
    )
    print(f"Max temperature during heat wave: {heat_wave['DryBulb'].max()}°C")
    
    # Сохраняем EPW
    generator.save_to_epw(heat_wave, "tokyo_heatwave_2050.epw")
    
    # Получаем граничные условия для CHT
    print("\n=== Граничные условия для CHT (полдень) ===")
    bc = generator.get_boundary_conditions(
        month=7, day=15, hour=12,
        t_min=25.0, t_max=33.0,
        cloud_cover=0.3, rh_avg=70.0
    )
    for key, value in bc.items():
        print(f"  {key}: {value}")
