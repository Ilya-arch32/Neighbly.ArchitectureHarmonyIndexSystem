# -*- coding: utf-8 -*-
# examples/beauty_analysis_example.py

# Демонстрационный пример использования модуля beauty_indexer.py
# для анализа эстетического качества помещений

import sys
import os

# Добавляем корневую директорию проекта в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.beauty_indexer import calculate_beauty_index, get_beauty_index_interpretation


def main():
    """
    Демонстрация анализа красоты для трех различных сценариев помещений.
    """
    print("=== АНАЛИЗ ЭСТЕТИЧЕСКОГО КАЧЕСТВА ПОМЕЩЕНИЙ ===")
    print("Модуль: beauty_indexer.py")
    print("Принцип: функция пропорциональна эстетика\n")
    
    # Сценарий 1: Идеальная гостиная в скандинавском стиле
    print("СЦЕНАРИЙ 1: Идеальная гостиная в скандинавском стиле")
    print("-" * 60)
    
    beauty_data_1 = {
        "colors_hex": ["#F5F5DC", "#8FBC8F", "#D2B48C"],  # Триадическая схема: бежевый, зеленый, коричневый
        "overall_style": "Скандинавский",
        "descriptions": [
            "диван из светлого дерева с белыми подушками",
            "журнальный столик из натурального дерева",
            "уютный текстильный плед"
        ],
        "room_area": 25.0,      # 25 м²
        "furniture_area": 10.0  # 40% занятости - идеально
    }
    
    results_1 = calculate_beauty_index(beauty_data_1)
    
    print(f"Цвета: {beauty_data_1['colors_hex']}")
    print(f"Стиль: {beauty_data_1['overall_style']}")
    print(f"Площадь комнаты: {beauty_data_1['room_area']} кв.м")
    print(f"Площадь мебели: {beauty_data_1['furniture_area']} кв.м")
    print()
    print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"• Цветовая гармония: {results_1['color_score']}")
    print(f"• Стилевое единство: {results_1['style_score']}")
    print(f"• Пространственная организация: {results_1['space_score']}")
    print(f"• ИТОГОВЫЙ ИНДЕКС КРАСОТЫ: {results_1['final_beauty_index']}")
    print(f"• Интерпретация: {get_beauty_index_interpretation(results_1['final_beauty_index'])}")
    print()
    
    # Сценарий 2: Проблемная комната с плохой цветовой схемой
    print("СЦЕНАРИЙ 2: Проблемная комната с несогласованным дизайном")
    print("-" * 60)
    
    beauty_data_2 = {
        "colors_hex": ["#FF0000", "#00FF00"],  # Комплементарные цвета, но слишком яркие
        "overall_style": "Минимализм",
        "descriptions": [
            "старый деревянный стол с резьбой",
            "яркий диван с узорами",
            "металлическая полка с декором"
        ],
        "room_area": 20.0,      # 20 м²
        "furniture_area": 16.0  # 80% занятости - перегружено
    }
    
    results_2 = calculate_beauty_index(beauty_data_2)
    
    print(f"Цвета: {beauty_data_2['colors_hex']}")
    print(f"Стиль: {beauty_data_2['overall_style']}")
    print(f"Площадь комнаты: {beauty_data_2['room_area']} кв.м")
    print(f"Площадь мебели: {beauty_data_2['furniture_area']} кв.м")
    print()
    print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"• Цветовая гармония: {results_2['color_score']}")
    print(f"• Стилевое единство: {results_2['style_score']}")
    print(f"• Пространственная организация: {results_2['space_score']}")
    print(f"• ИТОГОВЫЙ ИНДЕКС КРАСОТЫ: {results_2['final_beauty_index']}")
    print(f"• Интерпретация: {get_beauty_index_interpretation(results_2['final_beauty_index'])}")
    print()
    
    # Сценарий 3: Стильная лофт-студия
    print("СЦЕНАРИЙ 3: Стильная лофт-студия")
    print("-" * 60)
    
    beauty_data_3 = {
        "colors_hex": ["#8B4513", "#2F4F4F"],  # Аналоговая схема: коричневый и темно-серый
        "overall_style": "Лофт",
        "descriptions": [
            "стол из темного дерева и металла",
            "промышленный стул с металлическим каркасом",
            "грубая деревянная полка на металлических кронштейнах"
        ],
        "room_area": 30.0,      # 30 м²
        "furniture_area": 12.0  # 40% занятости - идеально
    }
    
    results_3 = calculate_beauty_index(beauty_data_3)
    
    print(f"Цвета: {beauty_data_3['colors_hex']}")
    print(f"Стиль: {beauty_data_3['overall_style']}")
    print(f"Площадь комнаты: {beauty_data_3['room_area']} кв.м")
    print(f"Площадь мебели: {beauty_data_3['furniture_area']} кв.м")
    print()
    print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"• Цветовая гармония: {results_3['color_score']}")
    print(f"• Стилевое единство: {results_3['style_score']}")
    print(f"• Пространственная организация: {results_3['space_score']}")
    print(f"• ИТОГОВЫЙ ИНДЕКС КРАСОТЫ: {results_3['final_beauty_index']}")
    print(f"• Интерпретация: {get_beauty_index_interpretation(results_3['final_beauty_index'])}")
    print()
    
    # Сравнительный анализ
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("-" * 60)
    print(f"Сценарий 1 (Скандинавский): {results_1['final_beauty_index']}")
    print(f"Сценарий 2 (Проблемный): {results_2['final_beauty_index']}")
    print(f"Сценарий 3 (Лофт): {results_3['final_beauty_index']}")
    print()
    print("ВЫВОДЫ:")
    print("• Цветовая гармония критически важна для общего восприятия")
    print("• Соответствие мебели заявленному стилю значительно влияет на оценку")
    print("• Правильная пространственная организация (40% занятости) оптимальна")
    print("• Модуль успешно выявляет эстетические проблемы и достоинства")


if __name__ == "__main__":
    main()
