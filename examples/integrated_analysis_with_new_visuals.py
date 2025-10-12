# examples/integrated_analysis_with_new_visuals.py

"""
Интегрированный анализ помещения с использованием новой системы визуализации.
Демонстрирует полный цикл: анализ -> расчет индексов -> создание отдельных диаграмм.
"""

import os
import sys

# Добавляем корневую папку проекта в путь
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from modules.thermal import calculate_total_heat_loss
from modules.optic import analyze_natural_light
from modules.indexer import calculate_function_index
from modules.beauty_indexer import calculate_beauty_index
from visualizer.passport import (
    create_harmony_gauge,
    create_subindex_chart,
    create_details_view,
    create_before_after_chart
)
from physipy import m, K

def main():
    """Выполняет полный анализ помещения с новой системой визуализации."""
    
    print("=== Интегрированный анализ помещения с новой визуализацией ===\n")
    
    # Параметры комнаты
    room_width = 5.0
    room_depth = 4.0
    room_height = 2.7
    room_area = room_width * room_depth * m**2
    
    # Температурные условия
    temp_inside = 22 * K
    temp_outside = -5 * K
    
    # Компоненты для термодинамического анализа
    room_components = [
        {"name": "Стены", "material": "brick_wall_uninsulated", "area": 40 * m**2},
        {"name": "Окно", "material": "double_pane_glass", "area": 3.0 * m**2}
    ]
    
    # Данные для оптического анализа
    room_data = {
        "room_dimensions": {
            "width": room_width,
            "depth": room_depth,
            "height": room_height
        },
        "windows": [
            {
                "width": 2.0,
                "height": 1.5,
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
    
    # Данные для анализа красоты
    beauty_data = {
        'colors_hex': ['#E8C547', '#6A5ACD', '#CD5C5C'],  # Золотой, фиолетовый, терракотовый
        'overall_style': 'Скандинавский',
        'descriptions': [
            'минималистичный деревянный стол',
            'светлый диван в скандинавском стиле',
            'простые полки из натурального дерева'
        ],
        'room_area': 20.0,
        'furniture_area': 8.0
    }
    
    print("1. Выполняем термодинамический анализ...")
    thermal_loss = calculate_total_heat_loss(room_components, temp_inside, temp_outside)
    
    print("2. Выполняем анализ естественного освещения...")
    optic_results = analyze_natural_light(room_data)
    
    print("3. Рассчитываем индекс функциональности...")
    function_results = calculate_function_index(thermal_loss, optic_results, room_area)
    
    print("4. Рассчитываем индекс красоты...")
    beauty_results = calculate_beauty_index(beauty_data)
    
    print("5. Рассчитываем общий индекс архитектурной гармонии...")
    # Веса: функция 60%, красота 40%
    harmony_index = function_results['final_function_index'] * 0.6 + beauty_results['final_beauty_index'] * 0.4
    
    print("6. Анализируем результаты...")
    # Создаем простые рекомендации на основе результатов
    recommendations = {
        'structural_changes': [
            {'description': 'Улучшить изоляцию стен', 'impact': 0.15},
            {'description': 'Установить тройное остекление', 'impact': 0.12}
        ],
        'easy_fixes': [
            {'description': 'Добавить зеркала для отражения света', 'impact': 0.08},
            {'description': 'Покрасить стены в светлые тона', 'impact': 0.06}
        ]
    }
    
    # Создаем папку для результатов
    output_dir = os.path.join(project_root, "integrated_analysis_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n7. Создаем визуализации...")
    
    # Создаем визуализации для русского и английского языков
    languages = ['ru', 'en']
    
    for lang in languages:
        lang_suffix = f"_{lang}" if lang == 'en' else ""
        print(f"\nСоздание визуализаций на языке: {lang.upper()}")
        
        # 7.1 Спидометр общей гармонии
        harmony_path = os.path.join(output_dir, f"harmony_index{lang_suffix}.png")
        create_harmony_gauge(harmony_index, harmony_path, lang=lang)
        
        # 7.2 Сравнение функции и красоты
        subindex_path = os.path.join(output_dir, f"function_vs_beauty{lang_suffix}.png")
        create_subindex_chart(function_results['final_function_index'], 
                             beauty_results['final_beauty_index'], 
                             subindex_path, lang=lang)
        
        # 7.3 Детальная разбивка показателей
        details = {
            'thermal_score': function_results['thermal_score'],
            'optic_score': function_results['optic_score'],
            'color_score': beauty_results['color_score'],
            'style_score': beauty_results['style_score'],
            'space_score': beauty_results['space_score']
        }
        details_path = os.path.join(output_dir, f"detailed_breakdown{lang_suffix}.png")
        create_details_view(details, details_path, lang=lang)
        
        # 7.4 Потенциал улучшений (до и после)
        before_scores = {
            'harmony': harmony_index,
            'function': function_results['final_function_index'],
            'beauty': beauty_results['final_beauty_index']
        }
        
        # Симулируем улучшения на основе рекомендаций
        after_scores = {
            'harmony': min(1.0, harmony_index + 0.25),
            'function': min(1.0, function_results['final_function_index'] + 0.30),
            'beauty': min(1.0, beauty_results['final_beauty_index'] + 0.15)
        }
        
        comparison_path = os.path.join(output_dir, f"improvement_potential{lang_suffix}.png")
        create_before_after_chart(before_scores, after_scores, comparison_path, lang=lang)
    
    # Выводим результаты анализа
    print(f"\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    print(f"Индекс Архитектурной Гармонии: {harmony_index:.3f}")
    print(f"- Индекс Функции: {function_results['final_function_index']:.3f}")
    print(f"  - Термодинамика: {function_results['thermal_score']:.3f}")
    print(f"  - Оптика: {function_results['optic_score']:.3f}")
    print(f"- Индекс Красоты: {beauty_results['final_beauty_index']:.3f}")
    print(f"  - Цветовая гармония: {beauty_results['color_score']:.3f}")
    print(f"  - Стилевое единство: {beauty_results['style_score']:.3f}")
    print(f"  - Пространственная организация: {beauty_results['space_score']:.3f}")
    
    print(f"\nТеплопотери: {thermal_loss} ({float(thermal_loss.value) / (room_width * room_depth):.1f} Вт/кв.м)")
    print(f"Покрытие дневным светом: {optic_results['daylight_coverage_percent']}%")
    print(f"Равномерность освещения: {optic_results['uniformity_index']:.3f}")
    print(f"Интенсивность освещения: {optic_results['intensity_index']:.3f}")
    
    print(f"\n=== СОЗДАННЫЕ ВИЗУАЛИЗАЦИИ ===")
    print(f"Папка: {output_dir}")
    print("Русские версии:")
    print("- harmony_index.png - Спидометр общей гармонии")
    print("- function_vs_beauty.png - Сравнение функции и красоты")
    print("- detailed_breakdown.png - Детальная разбивка показателей")
    print("- improvement_potential.png - Потенциал улучшений")
    print("Английские версии:")
    print("- harmony_index_en.png - Harmony gauge")
    print("- function_vs_beauty_en.png - Function vs Beauty comparison")
    print("- detailed_breakdown_en.png - Detailed metrics breakdown")
    print("- improvement_potential_en.png - Improvement potential")
    
    print("\n*** НОВЫЙ ПРОФЕССИОНАЛЬНЫЙ ДИЗАЙН ***")
    print("+ Теплый бежевый фон (#F9F5EC)")
    print("+ Сдержанная цветовая палитра")
    print("+ Заголовки выровнены по левому краю")
    print("+ Иконки меню в правом верхнем углу")
    print("+ Горизонтальные диаграммы вместо вертикальных")
    print("+ Стильные progress bar'ы без рамок")
    print("+ Легенды перенесены под заголовки")
    
    print(f"\n=== ГЛАВНЫЕ РЕКОМЕНДАЦИИ ===")
    if recommendations.get('structural_changes'):
        print("Структурные изменения:")
        for change in recommendations['structural_changes'][:3]:
            print(f"• {change['description']} (эффект: +{change['impact']:.2f})")
    
    if recommendations.get('easy_fixes'):
        print("Легкие исправления:")
        for fix in recommendations['easy_fixes'][:3]:
            print(f"• {fix['description']} (эффект: +{fix['impact']:.2f})")
    
    print("\nИнтегрированный анализ завершен!")

if __name__ == "__main__":
    main()
