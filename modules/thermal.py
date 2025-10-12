# modules/thermal.py

# Импортируем необходимые инструменты из библиотеки physipy
from physipy import m, K, Quantity, units

# Импортируем константы из централизованного файла
from core.constants import U_VALUES

# Получаем единицу измерения Ватт (W) из словаря units
W = units['W']

def calculate_total_heat_loss(room_components, temp_inside, temp_outside):
    """
    Рассчитывает общие теплопотери для комнаты, состоящей из нескольких компонентов.
    physipy следит за тем, чтобы все расчеты были корректны по размерности.
    """
    # Инициализируем общие потери как величину physipy с единицами Ватт
    total_loss = 0 * W
    temp_difference = temp_inside - temp_outside

    # Проходим по каждому компоненту комнаты (стены, окна и т.д.)
    for component in room_components:
        material = component["material"]
        area = component["area"]
        
        u_value = U_VALUES.get(material)
        
        if u_value:
            # Формула теплопотерь: U-value * Площадь * Разница температур
            component_loss = u_value * area * temp_difference
            total_loss += component_loss
            
    return total_loss