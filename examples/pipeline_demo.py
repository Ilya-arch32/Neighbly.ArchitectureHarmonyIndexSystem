"""
Демонстрация работы Pipeline Manager - главного управляющего скрипта проекта NeighblyHomePhysicsLab

Этот скрипт показывает полный цикл анализа помещения от исходных данных до финальных визуализаций.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_manager import run_analysis_pipeline, create_sample_room_data


def demo_standard_analysis():
    """
    Демонстрация стандартного анализа помещения.
    """
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ PIPELINE MANAGER")
    print("=" * 60)
    print("Тип анализа: STANDARD (данные из пользовательской таблицы)")
    print()
    
    # Создаем тестовые данные комнаты
    room_data = create_sample_room_data()
    
    print("Исходные данные комнаты:")
    print(f"- Размеры: {room_data['room_dimensions']['length']}x{room_data['room_dimensions']['width']}x{room_data['room_dimensions']['height']} м")
    print(f"- Окна: {len(room_data['windows'])} шт.")
    print(f"- Стены: {len(room_data['walls'])} шт.")
    print(f"- Стиль: {room_data['declared_style']}")
    print(f"- Температура: {room_data['interior_temperature']}°C внутри, {room_data['exterior_temperature']}°C снаружи")
    print()
    
    # Запускаем анализ
    print("Запуск полного анализа...")
    results = run_analysis_pipeline(room_data, 'standard')
    
    # Выводим результаты
    if results.get('_metadata', {}).get('status') == 'completed':
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("=" * 60)
        
        harmony_index = results.get('harmony_index', 0)
        function_index = results.get('function_scores', {}).get('final_function_index', 0)
        beauty_index = results.get('beauty_scores', {}).get('overall_index', 0)
        
        print(f"🎯 ОБЩИЙ ИНДЕКС ГАРМОНИИ: {harmony_index:.3f}")
        print(f"🔧 Индекс функциональности: {function_index:.3f}")
        print(f"🎭 Индекс красоты: {beauty_index:.3f}")
        print()
        
        # Детальная разбивка функциональности
        function_scores = results.get('function_scores', {})
        print("ДЕТАЛИЗАЦИЯ ФУНКЦИОНАЛЬНОСТИ:")
        print(f"- Термодинамический индекс: {function_scores.get('thermal_score', 0):.3f}")
        print(f"- Оптический индекс: {function_scores.get('optic_score', 0):.3f}")
        print()
        
        # Детальная разбивка красоты
        beauty_scores = results.get('beauty_scores', {})
        print("ДЕТАЛИЗАЦИЯ КРАСОТЫ:")
        print(f"- Цветовая гармония: {beauty_scores.get('color_score', 0):.3f}")
        print(f"- Стилевое единство: {beauty_scores.get('style_score', 0):.3f}")
        print(f"- Пространственная организация: {beauty_scores.get('space_score', 0):.3f}")
        print()
        
        # Рекомендации
        recommendations = results.get('recommendations', {})
        structural_count = len(recommendations.get('structural_changes', []))
        easy_count = len(recommendations.get('easy_fixes', []))
        
        print("РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
        print(f"- Структурные изменения: {structural_count}")
        print(f"- Простые исправления: {easy_count}")
        print()
        
        # Созданные файлы
        print("СОЗДАННЫЕ ВИЗУАЛИЗАЦИИ:")
        for viz_type, path in results.get('visualization_paths', {}).items():
            file_size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"- {viz_type}: {path} ({file_size} байт)")
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("Все визуализации сохранены в папке pipeline_results/")
        print("=" * 60)
        
    else:
        print("\nОШИБКА ПРИ АНАЛИЗЕ:")
        print(results.get('error_details', {}).get('user_message', 'Неизвестная ошибка'))


def demo_special_analysis():
    """
    Демонстрация специального анализа (данные от ИИ-аналитика).
    """
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ СПЕЦИАЛЬНОГО АНАЛИЗА")
    print("=" * 60)
    print("Тип анализа: SPECIAL (данные от ИИ-аналитика)")
    print()
    
    # Создаем данные, как будто они пришли от ИИ-аналитика
    ai_processed_data = {
        'room_dimensions': {'length': 6.0, 'width': 4.5, 'height': 3.0},
        'windows': [
            {'width': 2.0, 'height': 1.5, 'orientation': 'south', 'glass_type': 'double_glazing'}
        ],
        'walls': [
            {'area': 18.0, 'material': 'brick_wall_insulated', 'orientation': 'north'},
            {'area': 18.0, 'material': 'brick_wall_insulated', 'orientation': 'south'},
            {'area': 13.5, 'material': 'brick_wall_insulated', 'orientation': 'east'},
            {'area': 13.5, 'material': 'brick_wall_insulated', 'orientation': 'west'}
        ],
        'ceiling': {'area': 27.0, 'material': 'insulated_roof'},
        'floor': {'area': 27.0, 'material': 'concrete_slab'},
        'interior_temperature': 23.0,
        'exterior_temperature': -10.0,
        'surface_reflectances': {'walls': 0.8, 'ceiling': 0.9, 'floor': 0.4},
        'furniture_coverage': 0.3,
        'color_palette': ['#F5F5DC', '#8B4513', '#228B22'],
        'style_description': 'Современный скандинавский стиль с деревянными элементами',
        'declared_style': 'Скандинавский',
        '_ai_analysis_notes': 'Данные обработаны ИИ-аналитиком для оптимального результата'
    }
    
    print("Данные от ИИ-аналитика:")
    print(f"- Улучшенная изоляция стен: {ai_processed_data['walls'][0]['material']}")
    print(f"- Большие окна: {ai_processed_data['windows'][0]['width']}x{ai_processed_data['windows'][0]['height']} м")
    print(f"- Высокие коэффициенты отражения: {ai_processed_data['surface_reflectances']}")
    print()
    
    # Запускаем специальный анализ
    results = run_analysis_pipeline(ai_processed_data, 'special')
    
    if results.get('_metadata', {}).get('status') == 'completed':
        harmony_index = results.get('harmony_index', 0)
        print(f"РЕЗУЛЬТАТ СПЕЦИАЛЬНОГО АНАЛИЗА: {harmony_index:.3f}")
        print("Улучшенные параметры дали более высокий индекс гармонии!")
    else:
        print("Ошибка в специальном анализе")


if __name__ == "__main__":
    # Демонстрация стандартного анализа
    demo_standard_analysis()
    
    # Демонстрация специального анализа
    demo_special_analysis()
    
    print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("Pipeline Manager готов к использованию в реальных проектах.")
