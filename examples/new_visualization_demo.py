# examples/new_visualization_demo.py

"""
Демонстрация новой системы визуализации с отдельными изображениями.
Тестирует все 4 новые функции визуализации из visualizer/passport.py.
"""

import os
import sys

# Добавляем корневую папку проекта в путь
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from visualizer.passport import (
    create_harmony_gauge,
    create_subindex_chart, 
    create_details_view,
    create_before_after_chart
)

def main():
    """Демонстрирует все новые функции визуализации."""
    
    print("=== Демонстрация новой системы визуализации ===\n")
    
    # Создаем папку для результатов
    output_dir = os.path.join(project_root, "visualization_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Тестируем все функции для русского и английского языков
    languages = ['ru', 'en']
    
    for lang in languages:
        lang_suffix = f"_{lang}" if lang == 'en' else ""
        print(f"\n=== Создание визуализаций на языке: {lang.upper()} ===")
        
        # 1. Тестируем спидометр гармонии
        print(f"1. Создание спидометра Индекса Архитектурной Гармонии ({lang})...")
        harmony_index = 0.78  # Хороший показатель
        harmony_path = os.path.join(output_dir, f"harmony_gauge{lang_suffix}.png")
        create_harmony_gauge(harmony_index, harmony_path, lang=lang)
        
        # 2. Тестируем диаграмму составляющих
        print(f"2. Создание диаграммы составляющих гармонии ({lang})...")
        function_index = 0.65
        beauty_index = 0.82
        subindex_path = os.path.join(output_dir, f"subindices_chart{lang_suffix}.png")
        create_subindex_chart(function_index, beauty_index, subindex_path, lang=lang)
        
        # 3. Тестируем детальные показатели
        print(f"3. Создание детального анализа показателей ({lang})...")
        details = {
            'thermal_score': 0.72,
            'optic_score': 0.58,
            'color_score': 0.91,
            'style_score': 0.85,
            'space_score': 0.69
        }
        details_path = os.path.join(output_dir, f"details_view{lang_suffix}.png")
        create_details_view(details, details_path, lang=lang)
        
        # 4. Тестируем сравнение до/после
        print(f"4. Создание диаграммы сравнения 'До и После' ({lang})...")
        before_scores = {
            'harmony': 0.45,
            'function': 0.52,
            'beauty': 0.38
        }
        after_scores = {
            'harmony': 0.78,
            'function': 0.85,
            'beauty': 0.71
        }
        comparison_path = os.path.join(output_dir, f"before_after_chart{lang_suffix}.png")
        create_before_after_chart(before_scores, after_scores, comparison_path, lang=lang)
    
    print(f"\n=== Все визуализации созданы в папке: {output_dir} ===")
    print("\nСозданные файлы (русский и английский):")
    print("- harmony_gauge.png / harmony_gauge_en.png - Спидометр главного индекса")
    print("- subindices_chart.png / subindices_chart_en.png - Горизонтальная диаграмма функции и красоты")
    print("- details_view.png / details_view_en.png - Детальные показатели с progress bar'ами")
    print("- before_after_chart.png / before_after_chart_en.png - Сравнение результатов до и после улучшений")
    
    print("\n*** НОВЫЙ ДИЗАЙН ***")
    print("+ Теплый бежевый фон (#F9F5EC)")
    print("+ Профессиональная цветовая палитра")
    print("+ Заголовки выровнены по левому краю")
    print("+ Иконки меню в правом верхнем углу")
    print("+ Многоязычная поддержка (русский/английский)")
    print("+ Стильные progress bar'ы и сетки")
    
    print("\nДемонстрация завершена успешно!")

if __name__ == "__main__":
    main()
