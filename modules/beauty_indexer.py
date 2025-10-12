# modules/beauty_indexer.py

# Модуль для расчета "Временного Индекса Красоты" помещения
# на основе объективных критериев цветовой гармонии, стилевого единства и пространственной организации

import colorsys
import math


def calculate_beauty_index(beauty_data):
    """
    Рассчитывает итоговый индекс красоты помещения.
    
    Объединяет оценки цветовой гармонии, стилевого единства и пространственной организации
    в единый показатель эстетического качества помещения по принципу "функция ∝ эстетика".
    
    Args:
        beauty_data (dict): Словарь с данными для анализа красоты:
            - colors_hex (list): Список HEX-кодов цветов (2-3 цвета)
            - overall_style (str): Общий стиль помещения
            - descriptions (list): Описания предметов мебели
            - room_area (float): Общая площадь комнаты в м²
            - furniture_area (float): Площадь, занятая мебелью в м²
    
    Returns:
        dict: Словарь с индексами красоты
    """
    
    # Определяем веса для каждого показателя
    weights = {
        "color": 0.4,    # Цветовая гармония - 40%
        "style": 0.4,    # Стилевое единство - 40%
        "space": 0.2     # Пространственная организация - 20%
    }
    
    # Этап 1: Расчет оценки цветовой гармонии
    color_score = _calculate_color_score(beauty_data["colors_hex"])
    
    # Этап 2: Расчет оценки стилевого единства
    style_score = _calculate_style_score(
        beauty_data["overall_style"], 
        beauty_data["descriptions"]
    )
    
    # Этап 3: Расчет оценки пространственной организации
    space_score = _calculate_space_score(
        beauty_data["room_area"], 
        beauty_data["furniture_area"]
    )
    
    # Этап 4: Расчет итогового индекса красоты (средневзвешенное значение)
    final_beauty_index = (
        color_score * weights["color"] +
        style_score * weights["style"] +
        space_score * weights["space"]
    )
    
    return {
        "final_beauty_index": round(final_beauty_index, 2),
        "color_score": round(color_score, 2),
        "style_score": round(style_score, 2),
        "space_score": round(space_score, 2)
    }


def _calculate_color_score(colors_hex):
    """
    Рассчитывает оценку цветовой гармонии на основе HEX-кодов цветов.
    
    Анализирует цветовые схемы: комплементарную, аналоговую, триадическую.
    
    Args:
        colors_hex (list): Список HEX-кодов цветов (например, ['#6A5ACD', '#F0E68C'])
    
    Returns:
        float: Оценка цветовой гармонии от 0.0 до 1.0
    """
    if not colors_hex or len(colors_hex) < 2:
        return 0.5  # Базовая оценка при недостатке данных
    
    # Преобразуем HEX в HSL и извлекаем значения Hue в градусах
    hues = []
    for hex_color in colors_hex:
        hue_degrees = _hex_to_hue_degrees(hex_color)
        if hue_degrees is not None:
            hues.append(hue_degrees)
    
    if len(hues) < 2:
        return 0.5  # Базовая оценка при ошибке преобразования
    
    # Рассчитываем разности между всеми парами цветов
    differences = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            diff = min(abs(hues[i] - hues[j]), 360 - abs(hues[i] - hues[j]))
            differences.append(diff)
    
    # Проверяем на триадическую схему (3 цвета с разностью ~120°)
    if len(hues) == 3:
        triadic_diffs = [abs(diff - 120) for diff in differences]
        if all(diff <= 20 for diff in triadic_diffs):
            return 1.0  # Идеальная триадическая схема
    
    # Проверяем на комплементарную схему (разность ~180°)
    for diff in differences:
        if abs(diff - 180) <= 20:
            return 0.95  # Комплементарная схема
    
    # Проверяем на аналоговую схему (разность < 40°)
    if all(diff < 40 for diff in differences):
        return 0.9  # Аналоговая схема
    
    # В остальных случаях используем обратную зависимость от отклонения
    avg_diff = sum(differences) / len(differences)
    
    # Идеальные значения для оценки
    ideal_values = [30, 120, 180]  # Аналоговая, триадическая, комплементарная
    
    # Находим ближайшее идеальное значение
    min_deviation = min(abs(avg_diff - ideal) for ideal in ideal_values)
    
    # Базовая оценка с учетом отклонения
    score = max(0.0, 0.8 - (min_deviation / 90))  # Нормализуем отклонение
    
    return min(1.0, score)


def _calculate_style_score(overall_style, descriptions):
    """
    Рассчитывает оценку стилевого единства на основе соответствия мебели общему стилю.
    
    Args:
        overall_style (str): Название общего стиля помещения
        descriptions (list): Список описаний предметов мебели
    
    Returns:
        float: Оценка стилевого единства от 0.0 до 1.0
    """
    # Словарь ключевых слов для каждого стиля
    STYLE_KEYWORDS = {
        "Лофт": ["металл", "дерево", "кирпич", "грубый", "промышленный", "бетон"],
        "Скандинавский": ["светлое дерево", "белый", "простой", "уютный", "текстиль", "натуральный"],
        "Минимализм": ["простой", "геометрия", "нейтральный", "однотонный", "скрытый"],
        "Бохо": ["плетеный", "натуральный", "яркий", "узор", "растения", "текстиль"]
    }
    
    if not descriptions or overall_style not in STYLE_KEYWORDS:
        return 0.5  # Базовая оценка при недостатке данных
    
    keywords = STYLE_KEYWORDS[overall_style]
    matched_descriptions = 0
    
    # Проверяем каждое описание на наличие ключевых слов
    for description in descriptions:
        description_lower = description.lower()
        if any(keyword.lower() in description_lower for keyword in keywords):
            matched_descriptions += 1
    
    # Оценка = отношение совпавших описаний к общему количеству
    score = matched_descriptions / len(descriptions) if descriptions else 0.0
    
    return min(1.0, score)


def _calculate_space_score(room_area, furniture_area):
    """
    Рассчитывает оценку пространственной организации на основе коэффициента занятости.
    
    Args:
        room_area (float): Общая площадь комнаты в м²
        furniture_area (float): Площадь, занятая мебелью в м²
    
    Returns:
        float: Оценка пространственной организации от 0.0 до 1.0
    """
    if room_area <= 0 or furniture_area < 0:
        return 0.0  # Некорректные данные
    
    # Идеальный коэффициент занятости (40% площади занято мебелью)
    ideal_ratio = 0.4
    
    # Фактический коэффициент занятости
    actual_ratio = furniture_area / room_area
    
    # Оценка максимальна при ideal_ratio и снижается при отклонении
    score = max(0.0, 1.0 - abs(actual_ratio - ideal_ratio) / ideal_ratio)
    
    return min(1.0, score)


def _hex_to_hue_degrees(hex_color):
    """
    Утилита для преобразования HEX-кода в значение Hue в градусах.
    
    Args:
        hex_color (str): HEX-код цвета (например, '#6A5ACD')
    
    Returns:
        float: Значение Hue в градусах (0-360) или None при ошибке
    """
    try:
        # Убираем символ # если он есть
        hex_color = hex_color.lstrip('#')
        
        # Преобразуем HEX в RGB (0-1)
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        # Преобразуем RGB в HSL и извлекаем Hue
        hue, _, _ = colorsys.rgb_to_hls(r, g, b)
        
        # Преобразуем Hue из диапазона [0, 1] в градусы [0, 360]
        return hue * 360.0
        
    except (ValueError, IndexError):
        return None  # Ошибка преобразования


def get_beauty_index_interpretation(beauty_index):
    """
    Возвращает текстовую интерпретацию индекса красоты.
    
    Args:
        beauty_index (float): Индекс красоты от 0 до 1
        
    Returns:
        str: Текстовое описание эстетического качества помещения
    """
    if beauty_index >= 0.9:
        return "Превосходно - гармоничное и эстетически совершенное пространство"
    elif beauty_index >= 0.8:
        return "Отлично - красивое и стильное помещение с хорошей эстетикой"
    elif beauty_index >= 0.7:
        return "Хорошо - привлекательное помещение с приемлемой эстетикой"
    elif beauty_index >= 0.6:
        return "Удовлетворительно - помещение требует эстетических улучшений"
    elif beauty_index >= 0.4:
        return "Плохо - помещение имеет серьезные эстетические недостатки"
    else:
        return "Критично - помещение требует кардинального эстетического переосмысления"
