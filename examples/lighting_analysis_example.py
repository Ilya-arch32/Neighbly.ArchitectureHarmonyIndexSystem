# examples/lighting_analysis_example.py

import sys
import os

# Добавляем корневую директорию проекта в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.optic import analyze_natural_light

def main():
    """
    Демонстрация анализа естественного освещения комнаты.
    """
    
    print("=== Анализ естественного освещения комнаты ===\n")
    
    # Пример 1: Хорошо освещенная комната с южным окном
    room_data_good = {
        "room_dimensions": {
            "width": 5,      # метры
            "depth": 6,      # метры  
            "height": 2.8    # метры
        },
        "windows": [
            {
                "width": 2.5,           # метры
                "height": 1.8,          # метры
                "sill_height": 0.8,     # высота подоконника от пола в метрах
                "orientation": "south"   # южная ориентация
            }
        ],
        "surface_reflectance": {
            "walls": 0.7,    # светлые стены
            "floor": 0.3,    # средний пол
            "ceiling": 0.85  # светлый потолок
        },
        "location": {
            "latitude": 55.7  # Москва
        }
    }
    
    print("Пример 1: Комната с южным окном и светлыми стенами")
    print("Параметры:")
    print(f"  - Размеры: {room_data_good['room_dimensions']['width']}x{room_data_good['room_dimensions']['depth']}x{room_data_good['room_dimensions']['height']} м")
    print(f"  - Окно: {room_data_good['windows'][0]['width']}x{room_data_good['windows'][0]['height']} м, ориентация {room_data_good['windows'][0]['orientation']}")
    print(f"  - Отражение стен: {room_data_good['surface_reflectance']['walls']}")
    
    result_good = analyze_natural_light(room_data_good)
    
    print("\nРезультаты анализа:")
    print(f"  - Покрытие дневным светом: {result_good['daylight_coverage_percent']}%")
    print(f"  - Индекс равномерности: {result_good['uniformity_index']}")
    print(f"  - Индекс интенсивности: {result_good['intensity_index']}")
    
    if result_good['warnings']:
        print("  - Предупреждения:")
        for warning in result_good['warnings']:
            print(f"    • {warning}")
    else:
        print("  - Предупреждений нет")
    
    print("\n" + "="*60 + "\n")
    
    # Пример 2: Плохо освещенная комната с северным окном и темными стенами
    room_data_poor = {
        "room_dimensions": {
            "width": 4,      # метры
            "depth": 7,      # метры, глубокая комната
            "height": 2.5    # метры
        },
        "windows": [
            {
                "width": 1.5,           # небольшое окно
                "height": 1.2,          # низкое окно
                "sill_height": 1.0,     # высокий подоконник
                "orientation": "north"   # северная ориентация
            }
        ],
        "surface_reflectance": {
            "walls": 0.4,    # темные стены
            "floor": 0.2,    # темный пол
            "ceiling": 0.6   # средний потолок
        },
        "location": {
            "latitude": 60.0  # Санкт-Петербург (севернее)
        }
    }
    
    print("Пример 2: Комната с северным окном и темными стенами")
    print("Параметры:")
    print(f"  - Размеры: {room_data_poor['room_dimensions']['width']}x{room_data_poor['room_dimensions']['depth']}x{room_data_poor['room_dimensions']['height']} м")
    print(f"  - Окно: {room_data_poor['windows'][0]['width']}x{room_data_poor['windows'][0]['height']} м, ориентация {room_data_poor['windows'][0]['orientation']}")
    print(f"  - Отражение стен: {room_data_poor['surface_reflectance']['walls']}")
    
    result_poor = analyze_natural_light(room_data_poor)
    
    print("\nРезультаты анализа:")
    print(f"  - Покрытие дневным светом: {result_poor['daylight_coverage_percent']}%")
    print(f"  - Индекс равномерности: {result_poor['uniformity_index']}")
    print(f"  - Индекс интенсивности: {result_poor['intensity_index']}")
    
    if result_poor['warnings']:
        print("  - Предупреждения:")
        for warning in result_poor['warnings']:
            print(f"    • {warning}")
    else:
        print("  - Предупреждений нет")
    
    print("\n" + "="*60 + "\n")
    
    # Пример 3: Комната с несколькими окнами разной ориентации
    room_data_multi = {
        "room_dimensions": {
            "width": 6,      # метры
            "depth": 5,      # метры
            "height": 3.0    # метры
        },
        "windows": [
            {
                "width": 2.0,
                "height": 1.5,
                "sill_height": 0.9,
                "orientation": "south"
            },
            {
                "width": 1.8,
                "height": 1.5,
                "sill_height": 0.9,
                "orientation": "east"
            }
        ],
        "surface_reflectance": {
            "walls": 0.6,    # средне-светлые стены
            "floor": 0.35,   # средний пол
            "ceiling": 0.8   # светлый потолок
        },
        "location": {
            "latitude": 55.7  # Москва
        }
    }
    
    print("Пример 3: Комната с двумя окнами (южное и восточное)")
    print("Параметры:")
    print(f"  - Размеры: {room_data_multi['room_dimensions']['width']}x{room_data_multi['room_dimensions']['depth']}x{room_data_multi['room_dimensions']['height']} м")
    print(f"  - Окно 1: {room_data_multi['windows'][0]['width']}x{room_data_multi['windows'][0]['height']} м, ориентация {room_data_multi['windows'][0]['orientation']}")
    print(f"  - Окно 2: {room_data_multi['windows'][1]['width']}x{room_data_multi['windows'][1]['height']} м, ориентация {room_data_multi['windows'][1]['orientation']}")
    print(f"  - Отражение стен: {room_data_multi['surface_reflectance']['walls']}")
    
    result_multi = analyze_natural_light(room_data_multi)
    
    print("\nРезультаты анализа:")
    print(f"  - Покрытие дневным светом: {result_multi['daylight_coverage_percent']}%")
    print(f"  - Индекс равномерности: {result_multi['uniformity_index']}")
    print(f"  - Индекс интенсивности: {result_multi['intensity_index']}")
    
    if result_multi['warnings']:
        print("  - Предупреждения:")
        for warning in result_multi['warnings']:
            print(f"    • {warning}")
    else:
        print("  - Предупреждений нет")
    
    print("\n" + "="*60)
    print("\nАнализ завершен. Модуль optic.py готов к использованию!")

if __name__ == "__main__":
    main()
