# modules/indexer.py

# Модуль для расчета итогового индекса функциональности помещения
# на основе результатов термодинамического и оптического анализа

from physipy import m
from core.constants import BENCHMARKS

def calculate_function_index(thermal_results, optic_results, room_area):
    """
    Рассчитывает итоговый индекс функциональности помещения.
    
    Объединяет результаты термодинамического и оптического анализа
    в единый показатель качества помещения по принципу "функция ∝ эстетика".
    
    Args:
        thermal_results: Результат теплопотерь из thermal.py (физическая величина с единицами)
        optic_results (dict): Результаты анализа освещения из optic.py
        room_area: Площадь комнаты (физическая величина с единицами м²)
    
    Returns:
        dict: Словарь с индексами функциональности
    """
    
    # Извлекаем эталонные значения
    benchmarks = BENCHMARKS
    
    # Этап 1: Расчет термодинамического индекса (thermal_score)
    thermal_score = _calculate_thermal_score(thermal_results, room_area, benchmarks)
    
    # Этап 2: Расчет оптического индекса (optic_score)  
    optic_score = _calculate_optic_score(optic_results, benchmarks)
    
    # Этап 3: Расчет итогового индекса функциональности
    final_function_index = (thermal_score + optic_score) / 2.0
    
    return {
        "final_function_index": round(final_function_index, 2),
        "thermal_score": round(thermal_score, 2),
        "optic_score": round(optic_score, 2)
    }


def _calculate_thermal_score(thermal_results, room_area, benchmarks):
    """
    Рассчитывает термодинамический индекс (0-1) на основе теплопотерь.
    """
    # Преобразуем теплопотери в Вт/м² (удаляем единицы physipy для расчетов)
    total_loss_watts = float(thermal_results.value)  # Получаем числовое значение в Ваттах
    room_area_sqm = float(room_area.value)  # Получаем площадь в м²
    
    actual_loss_per_sqm = total_loss_watts / room_area_sqm
    ideal_loss_per_sqm = benchmarks["ideal_heat_loss_per_sqm"]
    
    # Логика оценки: идеальное значение = 0.95, вдвое худшее = 0.5
    # Используем экспоненциальную функцию для плавного перехода
    if actual_loss_per_sqm <= ideal_loss_per_sqm:
        # Если потери меньше или равны идеальным - высокая оценка
        thermal_score = 0.95 - (actual_loss_per_sqm / ideal_loss_per_sqm) * 0.05
    else:
        # Если потери больше идеальных - штрафуем экспоненциально
        ratio = actual_loss_per_sqm / ideal_loss_per_sqm
        thermal_score = 0.95 * (0.5 / 0.95) ** ((ratio - 1) / 1)  # При ratio=2 получаем ~0.5
    
    return max(0.0, min(1.0, thermal_score))  # Ограничиваем диапазон [0, 1]


def _calculate_optic_score(optic_results, benchmarks):
    """
    Рассчитывает оптический индекс (0-1) на основе показателей освещения.
    """
    # Извлекаем фактические значения
    actual_coverage = optic_results["daylight_coverage_percent"]
    actual_uniformity = optic_results["uniformity_index"]
    actual_intensity = optic_results["intensity_index"]
    
    # Извлекаем эталонные значения
    ideal_coverage = benchmarks["ideal_daylight_coverage"]
    ideal_uniformity = benchmarks["ideal_uniformity_index"]
    ideal_intensity = benchmarks["ideal_intensity_index"]
    
    # Извлекаем веса
    weights = benchmarks["optic_weights"]
    
    # Рассчитываем нормализованные оценки для каждого показателя
    coverage_score = min(1.0, actual_coverage / ideal_coverage)
    uniformity_score = min(1.0, actual_uniformity / ideal_uniformity)
    intensity_score = min(1.0, actual_intensity / ideal_intensity)
    
    # Вычисляем средневзвешенное значение
    optic_score = (
        coverage_score * weights["coverage"] +
        uniformity_score * weights["uniformity"] +
        intensity_score * weights["intensity"]
    )
    
    return max(0.0, min(1.0, optic_score))  # Ограничиваем диапазон [0, 1]


def get_function_index_interpretation(function_index):
    """
    Возвращает текстовую интерпретацию индекса функциональности.
    
    Args:
        function_index (float): Индекс функциональности от 0 до 1
        
    Returns:
        str: Текстовое описание качества помещения
    """
    if function_index >= 0.9:
        return "Превосходно - идеальное помещение с отличной функциональностью"
    elif function_index >= 0.8:
        return "Отлично - высококачественное помещение с хорошей функциональностью"
    elif function_index >= 0.7:
        return "Хорошо - качественное помещение с приемлемой функциональностью"
    elif function_index >= 0.6:
        return "Удовлетворительно - помещение требует улучшений"
    elif function_index >= 0.4:
        return "Плохо - помещение имеет серьезные недостатки"
    else:
        return "Критично - помещение непригодно для комфортного использования"
