# -*- coding: utf-8 -*-
# examples/beauty_module_test.py

# Простой тест для проверки отдельных функций модуля beauty_indexer.py

import sys
import os

# Добавляем корневую директорию проекта в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.beauty_indexer import (
    _calculate_color_score, 
    _calculate_style_score, 
    _calculate_space_score,
    _hex_to_hue_degrees
)


def test_color_score():
    """Тестирование функции расчета цветовой гармонии."""
    print("=== ТЕСТ ЦВЕТОВОЙ ГАРМОНИИ ===")
    
    # Триадическая схема (должна дать высокую оценку)
    triadic_colors = ["#FF0000", "#00FF00", "#0000FF"]  # Красный, зеленый, синий
    triadic_score = _calculate_color_score(triadic_colors)
    print(f"Триадическая схема {triadic_colors}: {triadic_score}")
    
    # Комплементарная схема
    complementary_colors = ["#FF0000", "#00FFFF"]  # Красный и циан
    complementary_score = _calculate_color_score(complementary_colors)
    print(f"Комплементарная схема {complementary_colors}: {complementary_score}")
    
    # Аналоговая схема
    analogous_colors = ["#FF0000", "#FF4500"]  # Красный и оранжево-красный
    analogous_score = _calculate_color_score(analogous_colors)
    print(f"Аналоговая схема {analogous_colors}: {analogous_score}")
    
    print()


def test_style_score():
    """Тестирование функции расчета стилевого единства."""
    print("=== ТЕСТ СТИЛЕВОГО ЕДИНСТВА ===")
    
    # Идеальное соответствие стилю Лофт
    loft_descriptions = [
        "стол из темного дерева и металла",
        "промышленный стул с металлическим каркасом",
        "грубая кирпичная стена"
    ]
    loft_score = _calculate_style_score("Лофт", loft_descriptions)
    print(f"Лофт (идеальное соответствие): {loft_score}")
    
    # Частичное соответствие стилю Скандинавский
    scandi_descriptions = [
        "диван из светлого дерева",
        "металлический стул",  # Не соответствует стилю
        "уютный текстильный плед"
    ]
    scandi_score = _calculate_style_score("Скандинавский", scandi_descriptions)
    print(f"Скандинавский (частичное соответствие): {scandi_score}")
    
    # Полное несоответствие
    mismatch_descriptions = [
        "пластиковый стол",
        "металлический стул",
        "синтетический ковер"
    ]
    mismatch_score = _calculate_style_score("Скандинавский", mismatch_descriptions)
    print(f"Скандинавский (несоответствие): {mismatch_score}")
    
    print()


def test_space_score():
    """Тестирование функции расчета пространственной организации."""
    print("=== ТЕСТ ПРОСТРАНСТВЕННОЙ ОРГАНИЗАЦИИ ===")
    
    # Идеальная занятость (40%)
    ideal_score = _calculate_space_score(25.0, 10.0)  # 40%
    print(f"Идеальная занятость (40%): {ideal_score}")
    
    # Перегруженность (80%)
    overloaded_score = _calculate_space_score(20.0, 16.0)  # 80%
    print(f"Перегруженность (80%): {overloaded_score}")
    
    # Пустота (10%)
    empty_score = _calculate_space_score(30.0, 3.0)  # 10%
    print(f"Пустота (10%): {empty_score}")
    
    print()


def test_hex_conversion():
    """Тестирование функции преобразования HEX в градусы."""
    print("=== ТЕСТ ПРЕОБРАЗОВАНИЯ HEX В ГРАДУСЫ ===")
    
    test_colors = {
        "#FF0000": "Красный (должен быть ~0°)",
        "#00FF00": "Зеленый (должен быть ~120°)",
        "#0000FF": "Синий (должен быть ~240°)",
        "#FFFF00": "Желтый (должен быть ~60°)"
    }
    
    for hex_color, description in test_colors.items():
        hue = _hex_to_hue_degrees(hex_color)
        print(f"{hex_color} - {description}: {hue:.1f}°")
    
    print()


def main():
    """Запуск всех тестов."""
    print("ТЕСТИРОВАНИЕ МОДУЛЯ BEAUTY_INDEXER.PY")
    print("=" * 50)
    print()
    
    test_color_score()
    test_style_score()
    test_space_score()
    test_hex_conversion()
    
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")


if __name__ == "__main__":
    main()
