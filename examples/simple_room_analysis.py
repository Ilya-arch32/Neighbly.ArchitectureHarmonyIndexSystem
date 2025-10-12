# -*- coding: utf-8 -*-
# examples/simple_room_analysis.py

# Комплексный анализ функциональности помещения
# Объединяет термодинамический и оптический анализ для расчета итогового индекса

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем все необходимые модули
from modules.thermal import calculate_total_heat_loss
from modules.optic import analyze_natural_light
from modules.indexer import calculate_function_index, get_function_index_interpretation
from modules.recommender import generate_recommendations
from visualizer.passport import create_room_passport, get_localized_interpretation

# Импортируем инструменты из physipy для определения величин с единицами
from physipy import m, K, Quantity, units

def main():
    """
    Демонстрация комплексного анализа функциональности помещения.
    """
    
    print("=== Комплексный анализ функциональности помещения ===\n")
    print("Философия проекта: функция пропорциональна эстетике\n")
    
    # --- Шаг 1: Определяем параметры комнаты ---
    
    # Размеры комнаты
    room_width = 5.0
    room_depth = 6.0
    room_height = 2.8
    room_area = room_width * room_depth * m**2
    
    print(f"Анализируемое помещение: {room_width}x{room_depth}x{room_height} м")
    print(f"Площадь: {float(room_area.value)} кв.м\n")
    
    # Температурные условия
    temp_inside = 22 * K
    temp_outside = -5 * K
    
    # Компоненты комнаты для термодинамического анализа
    room_components = [
        {"name": "Стены", "material": "brick_wall_uninsulated", "area": 40 * m**2},
        {"name": "Окно", "material": "double_pane_glass", "area": 4.5 * m**2}
    ]
    
    # Параметры комнаты для оптического анализа
    room_data = {
        "room_dimensions": {
            "width": room_width,
            "depth": room_depth,
            "height": room_height
        },
        "windows": [
            {
                "width": 2.5,
                "height": 1.8,
                "sill_height": 0.8,
                "orientation": "south"
            }
        ],
        "surface_reflectance": {
            "walls": 0.7,
            "floor": 0.3,
            "ceiling": 0.85
        },
        "location": {
            "latitude": 55.7  # Москва
        }
    }
    
    # --- Шаг 2: Термодинамический анализ ---
    
    print("--- ТЕРМОДИНАМИЧЕСКИЙ АНАЛИЗ ---")
    thermal_results = calculate_total_heat_loss(room_components, temp_inside, temp_outside)
    
    W = units['W']
    thermal_watts = thermal_results.to(W)
    thermal_per_sqm = float(thermal_results.value) / float(room_area.value)
    
    print(f"Общие теплопотери: {thermal_watts}")
    print(f"Теплопотери на кв.м: {thermal_per_sqm:.1f} Вт/кв.м")
    
    # --- Шаг 3: Оптический анализ ---
    
    print("\n--- ОПТИЧЕСКИЙ АНАЛИЗ ---")
    optic_results = analyze_natural_light(room_data)
    
    print(f"Покрытие дневным светом: {optic_results['daylight_coverage_percent']}%")
    print(f"Индекс равномерности: {optic_results['uniformity_index']}")
    print(f"Индекс интенсивности: {optic_results['intensity_index']}")
    
    if optic_results['warnings']:
        print("Предупреждения:")
        for warning in optic_results['warnings']:
            print(f"  • {warning}")
    
    # --- Шаг 4: Расчет итогового индекса функциональности ---
    
    print("\n--- ИНДЕКС ФУНКЦИОНАЛЬНОСТИ ---")
    function_results = calculate_function_index(thermal_results, optic_results, room_area)
    
    print(f"Термодинамический индекс: {function_results['thermal_score']}")
    print(f"Оптический индекс: {function_results['optic_score']}")
    print(f"ИТОГОВЫЙ ИНДЕКС ФУНКЦИИ: {function_results['final_function_index']}")
    
    # Интерпретация результата
    interpretation = get_function_index_interpretation(function_results['final_function_index'])
    print(f"\nОценка: {interpretation}")
    
    # --- Шаг 5: Интеллектуальные рекомендации ---
    
    target_index = 0.8  # Целевой индекс функциональности
    
    if function_results['final_function_index'] < target_index:
        print(f"\n--- ИНТЕЛЛЕКТУАЛЬНЫЙ АНАЛИЗ УЛУЧШЕНИЙ ---")
        print(f"Целевой индекс: {target_index}")
        print("Запускаю движок рекомендаций...\n")
        
        # Генерируем интеллектуальные рекомендации
        recommendations = generate_recommendations(room_data, function_results, target_index)
        
        # Выводим результаты
        _print_recommendations_report(recommendations)
        
    else:
        print("\n--- ПОЗДРАВЛЯЕМ! ---")
        print("• Помещение имеет отличную функциональность!")
        print("• Соблюден принцип 'функция пропорциональна эстетике'")
        print(f"• Достигнут целевой индекс {target_index}")
    
    # --- Шаг 6: Создание визуального паспорта ---
    
    print("\n--- СОЗДАНИЕ ВИЗУАЛЬНОГО ПАСПОРТА ---")
    print("Генерируется инфографика...")
    
    # Создаем визуальный паспорт помещения на русском языке
    create_room_passport(function_results, optic_results, interpretation, lang='ru')
    print("'Паспорт помещения' на русском сохранен в файл room_passport.png")
    
    # Создаем английскую версию
    interpretation_en = get_localized_interpretation(function_results['final_function_index'], lang='en')
    create_room_passport(function_results, optic_results, interpretation_en, lang='en')
    print("'Паспорт помещения' на английском сохранен в файл room_passport_en.png")
    
    print("\n" + "="*60)
    print("Анализ завершен. Все модули работают в интеграции!")


def _print_recommendations_report(recommendations):
    """
    Красиво выводит отчет с рекомендациями по улучшению помещения.
    """
    initial_scores = recommendations["initial_scores"]
    plan = recommendations["recommended_plan"]
    
    print(">> ТЕКУЩИЕ ПОКАЗАТЕЛИ:")
    print(f"   • Итоговый индекс функции: {initial_scores['final_function_index']}")
    print(f"   • Термодинамический индекс: {initial_scores['thermal_score']}")
    print(f"   • Оптический индекс: {initial_scores['optic_score']}")
    
    # Структурные рекомендации
    if plan["structural"]:
        print("\n>> СТРУКТУРНЫЕ ИЗМЕНЕНИЯ (High Impact):")
        for i, rec in enumerate(plan["structural"], 1):
            print(f"   {i}. {rec['recommendation']}")
            print(f"      Влияние: {rec['impact']}")
            print(f"      Новый индекс: {rec['final_score_after_change']}")
            print()
    
    # Легкие исправления
    if plan["lifestyle"]:
        print(">> ЛЕГКИЕ ИСПРАВЛЕНИЯ (Easy Fixes):")
        for i, rec in enumerate(plan["lifestyle"], 1):
            print(f"   {i}. {rec['recommendation']}")
            print(f"      Влияние: {rec['impact']}")
            print(f"      Новый индекс: {rec['final_score_after_change']}")
            print()
    
    # Итоговый совет
    print(">> РЕКОМЕНДАЦИЯ:")
    if plan["structural"] and plan["lifestyle"]:
        best_structural = max(plan["structural"], key=lambda x: x['final_score_after_change'])
        print(f"   Начните с: {best_structural['recommendation']}")
        print("   Затем примените легкие исправления для максимального эффекта.")
    elif plan["structural"]:
        print("   Сосредоточьтесь на структурных изменениях для достижения цели.")
    elif plan["lifestyle"]:
        print("   Используйте легкие исправления - они дадут заметный результат!")
    else:
        print("   Ваше помещение уже близко к идеалу!")


if __name__ == "__main__":
    main()