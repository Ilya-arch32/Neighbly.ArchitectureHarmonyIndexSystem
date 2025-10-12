# -*- coding: utf-8 -*-
# examples/localization_demo.py

# Демонстрация локализации визуализации "Паспорт Помещения"
# Показывает создание паспортов на русском и английском языках

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer.passport import get_localized_interpretation, TRANSLATIONS

def main():
    """
    Демонстрация функций локализации.
    """
    
    print("=== Демонстрация локализации 'Паспорт Помещения' ===\n")
    
    # Тестовый индекс функциональности
    test_index = 0.51
    
    print(f"Тестовый индекс функциональности: {test_index}")
    print()
    
    # Демонстрация переводов интерпретаций
    print("--- ИНТЕРПРЕТАЦИИ ---")
    interpretation_ru = get_localized_interpretation(test_index, lang='ru')
    interpretation_en = get_localized_interpretation(test_index, lang='en')
    
    print(f"Русский: {interpretation_ru}")
    print(f"English: {interpretation_en}")
    print()
    
    # Демонстрация переводов элементов интерфейса
    print("--- ПЕРЕВОДЫ ЭЛЕМЕНТОВ ИНТЕРФЕЙСА ---")
    
    interface_elements = [
        'function_index',
        'index_components', 
        'thermodynamics',
        'optics',
        'optics_details',
        'coverage',
        'uniformity',
        'intensity',
        'recommendations',
        'no_problems'
    ]
    
    print("Элемент интерфейса | Русский | English")
    print("-" * 60)
    
    for element in interface_elements:
        ru_text = TRANSLATIONS['ru'][element]
        en_text = TRANSLATIONS['en'][element]
        print(f"{element:<18} | {ru_text:<15} | {en_text}")
    
    print()
    print("--- РАЗЛИЧНЫЕ УРОВНИ ИНДЕКСА ---")
    
    test_indices = [0.95, 0.75, 0.55, 0.35]
    
    for idx in test_indices:
        ru_interp = get_localized_interpretation(idx, lang='ru')
        en_interp = get_localized_interpretation(idx, lang='en')
        print(f"Индекс {idx}: RU - {ru_interp}")
        print(f"Index {idx}: EN - {en_interp}")
        print()
    
    print("="*60)
    print("Локализация готова к использованию!")
    print("Для создания английской версии используйте: create_room_passport(..., lang='en')")

if __name__ == "__main__":
    main()
