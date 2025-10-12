# modules/optic.py

# Импортируем необходимые инструменты из библиотеки physipy
from physipy import m, Quantity, units
import math

# Импортируем константы из централизованного файла
from core.constants import ORIENTATION_MULTIPLIERS

def analyze_natural_light(room_data):
    """
    Анализирует качество естественного освещения в помещении.
    
    Принимает словарь с параметрами комнаты и возвращает структурированный
    анализ освещения с тремя ключевыми показателями.
    
    Args:
        room_data (dict): Словарь с параметрами комнаты, включающий:
            - room_dimensions: размеры комнаты (width, depth, height в метрах)
            - windows: список окон с их параметрами
            - surface_reflectance: коэффициенты отражения поверхностей
            - location: географические данные (latitude)
    
    Returns:
        dict: Результаты анализа освещения
    """
    
    # Извлекаем данные из входного словаря
    dimensions = room_data["room_dimensions"]
    windows = room_data["windows"]
    reflectance = room_data["surface_reflectance"]
    location = room_data["location"]
    
    # Преобразуем размеры в физические величины с единицами
    room_width = dimensions["width"] * m
    room_depth = dimensions["depth"] * m
    room_height = dimensions["height"] * m
    room_area = room_width * room_depth
    
    latitude = location["latitude"]
    
    # Этап А: Анализ прямого света
    direct_light_area, orientation_factors = _calculate_direct_light_coverage(
        windows, room_width, room_depth, room_height, latitude
    )
    
    # Этап Б: Анализ отраженного света
    reflected_light_quality = _calculate_reflected_light_quality(
        direct_light_area, room_area, reflectance, windows
    )
    
    # Этап В: Расчет итоговых метрик
    metrics = _calculate_final_metrics(
        direct_light_area, reflected_light_quality, room_area, 
        orientation_factors, reflectance
    )
    
    # Генерация предупреждений
    warnings = _generate_warnings(metrics, orientation_factors, reflectance)
    
    return {
        "daylight_coverage_percent": round(metrics["coverage"], 1),
        "uniformity_index": round(metrics["uniformity"], 2),
        "intensity_index": round(metrics["intensity"], 2),
        "warnings": warnings
    }


def _calculate_direct_light_coverage(windows, room_width, room_depth, room_height, latitude):
    """
    Этап А: Рассчитывает площадь пола, освещенную прямым солнечным светом.
    """
    total_direct_area = 0 * m**2
    orientation_factors = {}
    
    # Коррекция на широту (чем севернее, тем меньше прямого света)
    latitude_factor = max(0.5, 1.0 - abs(latitude - 45) / 90)
    
    for window in windows:
        window_width = window["width"] * m
        window_height = window["height"] * m
        orientation = window["orientation"]
        
        # Базовое правило: эффективный свет проникает на 2.25 высоты окна
        base_penetration = 2.25 * window_height
        
        # Применяем коррекции
        orientation_mult = ORIENTATION_MULTIPLIERS.get(orientation, 0.8)
        effective_penetration = base_penetration * orientation_mult * latitude_factor
        
        # Ограничиваем проникновение размерами комнаты
        max_penetration = min(effective_penetration, room_depth)
        
        # Площадь прямого освещения от этого окна
        window_direct_area = window_width * max_penetration
        total_direct_area += window_direct_area
        
        orientation_factors[orientation] = orientation_factors.get(orientation, 0) + 1
    
    return total_direct_area, orientation_factors


def _calculate_reflected_light_quality(direct_light_area, room_area, reflectance, windows):
    """
    Этап Б: Оценивает качество отраженного света в зонах без прямого освещения.
    """
    # Площадь без прямого света
    indirect_area = room_area - direct_light_area
    
    # Средний коэффициент отражения в комнате
    avg_reflectance = (
        reflectance["walls"] * 0.6 +  # Стены имеют больший вес
        reflectance["ceiling"] * 0.3 +
        reflectance["floor"] * 0.1
    )
    
    # Общая площадь окон для оценки количества входящего света
    total_window_area = sum(
        window["width"] * window["height"] for window in windows
    ) * m**2
    
    # Коэффициент отраженного света зависит от:
    # 1. Количества прямого света (больше прямого = больше отраженного)
    # 2. Коэффициентов отражения поверхностей
    # 3. Отношения площади окон к площади комнаты
    
    window_to_room_ratio = float(total_window_area / room_area)
    direct_light_ratio = float(direct_light_area / room_area)
    
    reflected_quality = avg_reflectance * (0.7 * direct_light_ratio + 0.3 * window_to_room_ratio)
    
    return {
        "quality": reflected_quality,
        "indirect_area_ratio": float(indirect_area / room_area),
        "avg_reflectance": avg_reflectance
    }


def _calculate_final_metrics(direct_light_area, reflected_light_quality, room_area, 
                           orientation_factors, reflectance):
    """
    Этап В: Вычисляет три ключевых показателя освещения.
    """
    # 1. Процент покрытия дневным светом
    direct_coverage = float(direct_light_area / room_area) * 100
    
    # Добавляем вклад отраженного света (до 30% дополнительного покрытия)
    reflected_contribution = reflected_light_quality["quality"] * 30
    total_coverage = min(100, direct_coverage + reflected_contribution)
    
    # 2. Индекс равномерности освещения
    # Зависит от распределения окон и качества отраженного света
    base_uniformity = reflected_light_quality["quality"]
    
    # Штраф за слишком большую долю площади без прямого света
    indirect_penalty = reflected_light_quality["indirect_area_ratio"] * 0.3
    uniformity = max(0, base_uniformity - indirect_penalty)
    
    # 3. Общий индекс интенсивности света
    # Комбинирует прямой и отраженный свет
    direct_intensity = min(1.0, direct_coverage / 60)  # 60% покрытия = максимум
    reflected_intensity = reflected_light_quality["quality"]
    
    intensity = (direct_intensity * 0.7 + reflected_intensity * 0.3)
    
    return {
        "coverage": total_coverage,
        "uniformity": uniformity,
        "intensity": intensity
    }


def _generate_warnings(metrics, orientation_factors, reflectance):
    """
    Генерирует предупреждения на основе анализа освещения.
    """
    warnings = []
    
    # Предупреждения о низком покрытии
    if metrics["coverage"] < 50:
        warnings.append("Недостаточное покрытие дневным светом. Рекомендуется увеличить размер окон.")
    
    # Предупреждения о неравномерности
    if metrics["uniformity"] < 0.4:
        warnings.append("Обнаружена потенциально темная зона в дальнем углу от окна.")
    
    # Предупреждения о низкой интенсивности
    if metrics["intensity"] < 0.5:
        warnings.append("Низкая общая интенсивность освещения. Рассмотрите возможность добавления окон.")
    
    # Предупреждения о темных поверхностях
    if reflectance["walls"] < 0.5:
        warnings.append("Темные стены снижают качество отраженного света. Рекомендуются светлые тона.")
    
    # Предупреждения об ориентации окон
    if "north" in orientation_factors and len(orientation_factors) == 1:
        warnings.append("Только северные окна обеспечивают ограниченное естественное освещение.")
    
    return warnings