# modules/recommender.py

# Интеллектуальный движок рекомендаций для оптимизации функциональности помещений
# Действует как виртуальный архитектор-консультант, предлагающий конкретные пути улучшения

import copy
from physipy import m, K
from modules.thermal import calculate_total_heat_loss
from modules.optic import analyze_natural_light
from modules.indexer import calculate_function_index

def generate_recommendations(initial_room_data, initial_analysis_results, target_index=0.85):
    """
    Генерирует интеллектуальные рекомендации по улучшению функциональности помещения.
    
    Проводит "что, если...?" анализ различных изменений и находит наиболее эффективные
    пути достижения целевого индекса функциональности.
    
    Args:
        initial_room_data (dict): Исходные параметры комнаты
        initial_analysis_results (dict): Результаты первичного анализа из indexer.py
        target_index (float): Целевое значение индекса (по умолчанию 0.85)
    
    Returns:
        dict: Подробный план рекомендаций с расчетом влияния каждого изменения
    """
    
    # Извлекаем начальные показатели
    initial_scores = {
        "final_function_index": initial_analysis_results["final_function_index"],
        "thermal_score": initial_analysis_results["thermal_score"],
        "optic_score": initial_analysis_results["optic_score"]
    }
    
    # Если индекс уже достигнут, возвращаем поздравления
    if initial_scores["final_function_index"] >= target_index:
        return {
            "initial_scores": initial_scores,
            "recommended_plan": {
                "structural": [],
                "lifestyle": []
            },
            "message": f"Поздравляем! Ваше помещение уже достигло целевого индекса {target_index}"
        }
    
    # Шаг 1: Определение "рычагов влияния"
    structural_levers = _get_structural_levers()
    lifestyle_levers = _get_lifestyle_levers()
    
    # Шаг 2: Цикл структурной оптимизации
    print("Анализирую структурные изменения...")
    structural_recommendations = _optimize_structural_changes(
        initial_room_data, structural_levers, initial_scores
    )
    
    # Шаг 3: Цикл оптимизации "Легких исправлений"
    print("Анализирую легкие исправления...")
    lifestyle_recommendations = _optimize_lifestyle_changes(
        initial_room_data, lifestyle_levers, initial_scores
    )
    
    # Шаг 4: Формирование итогового отчета
    return {
        "initial_scores": initial_scores,
        "recommended_plan": {
            "structural": structural_recommendations,
            "lifestyle": lifestyle_recommendations
        }
    }


def _get_structural_levers():
    """
    Возвращает список структурных параметров, которые можно изменять.
    """
    return [
        {
            "name": "window_width_increase",
            "description": "Увеличение ширины окна",
            "test_changes": [0.1, 0.2, 0.3, 0.5],  # Увеличение на 10%, 20%, 30%, 50%
            "max_change": 0.5
        },
        {
            "name": "window_height_increase", 
            "description": "Увеличение высоты окна",
            "test_changes": [0.1, 0.2, 0.3, 0.4],
            "max_change": 0.4
        },
        {
            "name": "add_window",
            "description": "Добавление дополнительного окна",
            "test_changes": [1],  # Добавить одно окно
            "max_change": 1
        },
        {
            "name": "improve_wall_material",
            "description": "Улучшение теплоизоляции стен",
            "test_changes": [1],  # Замена материала
            "max_change": 1
        }
    ]


def _get_lifestyle_levers():
    """
    Возвращает список "легких" параметров, которые можно изменять.
    """
    return [
        {
            "name": "wall_reflectance",
            "description": "Изменение отражательной способности стен",
            "test_values": [0.8, 0.85, 0.9],  # Светлые тона
            "current_param": "surface_reflectances.walls"
        },
        {
            "name": "ceiling_reflectance",
            "description": "Изменение отражательной способности потолка",
            "test_values": [0.9, 0.95],
            "current_param": "surface_reflectances.ceiling"
        },
        {
            "name": "add_mirror",
            "description": "Добавление зеркала для увеличения освещенности",
            "test_changes": [0.05, 0.08, 0.12],  # Прямое увеличение оптического индекса
            "max_change": 0.12
        }
    ]


def _optimize_structural_changes(initial_room_data, structural_levers, initial_scores):
    """
    Оптимизирует структурные изменения и находит наиболее эффективные.
    """
    best_recommendations = []
    
    for lever in structural_levers:
        best_change = None
        best_impact = 0
        
        for change_value in lever["test_changes"]:
            # Создаем копию данных комнаты для тестирования
            test_room_data = copy.deepcopy(initial_room_data)
            
            # Применяем тестовое изменение
            modified_data = _apply_structural_change(test_room_data, lever["name"], change_value)
            
            if modified_data is None:
                continue
                
            # Проводим полный анализ с измененными данными
            try:
                new_analysis = _run_full_analysis(modified_data)
                impact = new_analysis["final_function_index"] - initial_scores["final_function_index"]
                
                # Сохраняем лучшее изменение для этого рычага
                if impact > best_impact:
                    best_impact = impact
                    best_change = {
                        "change_value": change_value,
                        "new_score": new_analysis["final_function_index"],
                        "impact": impact
                    }
                    
            except Exception as e:
                print(f"Ошибка при тестировании {lever['name']}: {e}")
                continue
        
        # Добавляем лучшую рекомендацию для этого рычага
        if best_change and best_impact > 0.02:  # Минимальный порог улучшения
            recommendation = _format_structural_recommendation(
                lever, best_change, initial_room_data
            )
            best_recommendations.append(recommendation)
    
    # Сортируем по влиянию и возвращаем топ-2
    best_recommendations.sort(key=lambda x: float(x["impact"].replace("+", "").split()[0]), reverse=True)
    return best_recommendations[:2]


def _optimize_lifestyle_changes(initial_room_data, lifestyle_levers, initial_scores):
    """
    Оптимизирует легкие изменения и находит наиболее эффективные.
    """
    best_recommendations = []
    
    for lever in lifestyle_levers:
        best_change = None
        best_impact = 0
        
        if lever["name"] == "add_mirror":
            # Специальная логика для зеркала
            for impact_value in lever["test_changes"]:
                recommendation = {
                    "recommendation": f"Добавить большое зеркало (1.5м x 1.0м) на стену напротив окна",
                    "impact": f"+{impact_value:.2f} к Индексу Оптики",
                    "final_score_after_change": round(initial_scores["final_function_index"] + impact_value/2, 2)
                }
                best_recommendations.append(recommendation)
                break  # Берем первый вариант
        else:
            # Тестируем изменения отражательной способности
            for test_value in lever["test_values"]:
                test_room_data = copy.deepcopy(initial_room_data)
                
                # Применяем изменение
                if lever["name"] == "wall_reflectance":
                    test_room_data["surface_reflectances"]["walls"] = test_value
                elif lever["name"] == "ceiling_reflectance":
                    test_room_data["surface_reflectances"]["ceiling"] = test_value
                
                try:
                    new_analysis = _run_full_analysis(test_room_data)
                    impact = new_analysis["final_function_index"] - initial_scores["final_function_index"]
                    
                    if impact > best_impact:
                        best_impact = impact
                        best_change = {
                            "test_value": test_value,
                            "new_score": new_analysis["final_function_index"],
                            "impact": impact
                        }
                        
                except Exception as e:
                    continue
            
            # Добавляем лучшую рекомендацию
            if best_change and best_impact > 0.01:
                recommendation = _format_lifestyle_recommendation(
                    lever, best_change, initial_room_data
                )
                best_recommendations.append(recommendation)
    
    # Сортируем по влиянию и возвращаем топ-3
    best_recommendations.sort(key=lambda x: float(x["impact"].replace("+", "").split()[0]), reverse=True)
    return best_recommendations[:3]


def _apply_structural_change(room_data, change_name, change_value):
    """
    Применяет структурное изменение к данным комнаты.
    """
    try:
        if change_name == "window_width_increase":
            current_width = room_data["windows"][0]["width"]
            room_data["windows"][0]["width"] = current_width * (1 + change_value)
            
        elif change_name == "window_height_increase":
            current_height = room_data["windows"][0]["height"]
            room_data["windows"][0]["height"] = current_height * (1 + change_value)
            
        elif change_name == "add_window":
            # Добавляем второе окно (восточная ориентация)
            new_window = {
                "width": 1.5,
                "height": 1.2,
                "sill_height": 0.8,
                "orientation": "east"
            }
            room_data["windows"].append(new_window)
            
        elif change_name == "improve_wall_material":
            # Эта логика будет обрабатываться в термическом анализе
            room_data["wall_material_improved"] = True
            
        return room_data
        
    except Exception as e:
        print(f"Ошибка применения изменения {change_name}: {e}")
        return None


def _run_full_analysis(room_data):
    """
    Проводит полный анализ помещения с новыми параметрами.
    """
    # Извлекаем размеры для термического анализа
    dimensions = room_data["room_dimensions"]
    room_area = dimensions["width"] * dimensions["depth"] * m**2
    
    # Термический анализ
    temp_inside = 22 * K
    temp_outside = -5 * K
    
    # Определяем материал стен
    wall_material = "brick_wall_insulated" if room_data.get("wall_material_improved") else "brick_wall_uninsulated"
    
    # Рассчитываем общую площадь окон
    total_window_area = 0
    for window in room_data["windows"]:
        total_window_area += window["width"] * window["height"]
    
    room_components = [
        {"name": "Стены", "material": wall_material, "area": 40 * m**2},
        {"name": "Окно", "material": "double_pane_glass", "area": total_window_area * m**2}
    ]
    
    thermal_results = calculate_total_heat_loss(room_components, temp_inside, temp_outside)
    
    # Оптический анализ
    optic_results = analyze_natural_light(room_data)
    
    # Расчет индекса функциональности
    function_results = calculate_function_index(thermal_results, optic_results, room_area)
    
    return function_results


def _format_structural_recommendation(lever, best_change, initial_room_data):
    """
    Форматирует структурную рекомендацию в читаемый вид.
    """
    change_value = best_change["change_value"]
    impact = best_change["impact"]
    
    if lever["name"] == "window_width_increase":
        current_width = initial_room_data["windows"][0]["width"]
        new_width = current_width * (1 + change_value)
        return {
            "recommendation": f"Увеличить ширину окна на {int(change_value*100)}% (с {current_width}м до {new_width:.1f}м)",
            "impact": f"+{impact:.2f} к Итоговому Индексу",
            "final_score_after_change": best_change["new_score"]
        }
        
    elif lever["name"] == "window_height_increase":
        current_height = initial_room_data["windows"][0]["height"]
        new_height = current_height * (1 + change_value)
        return {
            "recommendation": f"Увеличить высоту окна на {int(change_value*100)}% (с {current_height}м до {new_height:.1f}м)",
            "impact": f"+{impact:.2f} к Итоговому Индексу",
            "final_score_after_change": best_change["new_score"]
        }
        
    elif lever["name"] == "add_window":
        return {
            "recommendation": "Добавить дополнительное окно (1.5м x 1.2м) с восточной ориентацией",
            "impact": f"+{impact:.2f} к Итоговому Индексу",
            "final_score_after_change": best_change["new_score"]
        }
        
    elif lever["name"] == "improve_wall_material":
        return {
            "recommendation": "Улучшить теплоизоляцию стен (замена на изолированные материалы)",
            "impact": f"+{impact:.2f} к Термодинамическому Индексу",
            "final_score_after_change": best_change["new_score"]
        }


def _format_lifestyle_recommendation(lever, best_change, initial_room_data):
    """
    Форматирует рекомендацию по легким изменениям в читаемый вид.
    """
    test_value = best_change["test_value"]
    impact = best_change["impact"]
    
    if lever["name"] == "wall_reflectance":
        current_value = initial_room_data["surface_reflectances"]["walls"]
        return {
            "recommendation": f"Перекрасить стены в более светлый цвет (коэфф. отражения {test_value})",
            "impact": f"+{impact:.2f} к Оптическому Индексу",
            "final_score_after_change": best_change["new_score"]
        }
        
    elif lever["name"] == "ceiling_reflectance":
        return {
            "recommendation": f"Использовать более светлый потолок (коэфф. отражения {test_value})",
            "impact": f"+{impact:.2f} к Оптическому Индексу", 
            "final_score_after_change": best_change["new_score"]
        }
