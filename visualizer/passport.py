# visualizer/passport.py

# Модуль для создания визуальных диаграмм анализа помещений
# Использует библиотеку Pillow для создания отдельных изображений

from PIL import Image, ImageDraw, ImageFont
import math
import os

# Глобальная константа для высокого разрешения
SCALE_FACTOR = 5

# Новая профессиональная цветовая палитра
PALETTE = {
    'background': '#F9F5EC',    # Теплый светло-бежевый фон
    'text_main': '#3D405B',     # Темный серо-синий для текста
    'text_light': '#A0A0A0',    # Светло-серый для меток
    'accent_yellow': '#E0C56B', # Горчично-желтый
    'accent_teal': '#3F888F',   # Бирюзово-зеленый
    'accent_orange': '#D48C63', # Терракотово-оранжевый
    'grid': '#EAEAEA'           # Очень светлый серый для сеток
}

# Словари для многоязычности
TRANSLATIONS = {
    'ru': {
        'harmony_gauge_title': 'Индекс Архитектурной Гармонии',
        'subindex_chart_title': 'Составляющие Гармонии',
        'details_view_title': 'Детальные Показатели',
        'before_after_title': 'Результаты Улучшений: До и После',
        'function_index': 'Индекс Функции',
        'beauty_index': 'Индекс Красоты',
        'function': 'Функция',
        'beauty': 'Красота',
        'thermodynamics': 'Термодинамика',
        'optics': 'Оптика',
        'color_harmony': 'Цветовая гармония',
        'style_unity': 'Стилевое единство',
        'space': 'Пространство',
        'harmony': 'Гармония',
        'poor': 'Плохо',
        'excellent': 'Отлично',
        'before': 'До',
        'after': 'После'
    },
    'en': {
        'harmony_gauge_title': 'Architectural Harmony Index',
        'subindex_chart_title': 'Harmony Components',
        'details_view_title': 'Detailed Metrics',
        'before_after_title': 'Improvements: Before & After',
        'function_index': 'Function Index',
        'beauty_index': 'Beauty Index',
        'function': 'Function',
        'beauty': 'Beauty',
        'thermodynamics': 'Thermodynamics',
        'optics': 'Optics',
        'color_harmony': 'Color Harmony',
        'style_unity': 'Style Unity',
        'space': 'Space',
        'harmony': 'Harmony',
        'poor': 'Poor',
        'excellent': 'Excellent',
        'before': 'Before',
        'after': 'After'
    }
}

def _load_font(size=20, bold=False, lang='ru'):
    """
    Загружает шрифт в зависимости от языка.
    
    Args:
        size (int): Размер шрифта
        bold (bool): Использовать жирный шрифт
        lang (str): Язык ('ru' или 'en')
        
    Returns:
        ImageFont: Объект шрифта
    """
    try:
        if lang == 'ru':
            # Для русского языка используем Google Sans
            font_path = "GoogleFonts/GoogleSans-Regular.ttf"
            if bold:
                font_path = "GoogleFonts/GoogleSans-Bold.ttf"
        else:
            # Для английского языка используем Canela
            font_path = "Canela-Thin-Trial.otf"
        
        return ImageFont.truetype(font_path, size)
    except:
        # Если не удается загрузить, используем системный шрифт
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()


def create_harmony_gauge(harmony_index: float, output_path: str, lang: str = 'ru'):
    """
    Создает элегантную круговую диаграмму для главного Индекса Архитектурной Гармонии.
    
    Args:
        harmony_index (float): Значение индекса от 0 до 1
        output_path (str): Путь для сохранения изображения
        lang (str): Язык ('ru' или 'en')
    """
    # Создаем квадратный холст с высоким разрешением
    img = Image.new('RGB', (500 * SCALE_FACTOR, 500 * SCALE_FACTOR), PALETTE['background'])
    draw = ImageDraw.Draw(img)
    
    # Загружаем шрифты с учетом языка и масштабирования
    value_font = _load_font(80 * SCALE_FACTOR, bold=True, lang=lang)
    label_font = _load_font(20 * SCALE_FACTOR, lang=lang)
    
    # Параметры круговой диаграммы с увеличенным радиусом
    center_x, center_y = 250 * SCALE_FACTOR, 250 * SCALE_FACTOR
    radius = 220 * SCALE_FACTOR  # Увеличен с 120 до 220
    ring_width = 15 * SCALE_FACTOR
    
    # Рисуем фоновое кольцо (полная окружность)
    draw.arc([(center_x - radius, center_y - radius), 
              (center_x + radius, center_y + radius)], 
             start=0, end=360, fill=PALETTE['grid'], width=ring_width)
    
    # Создаем градиентную активную дугу
    end_angle = harmony_index * 360
    
    # Рисуем активную дугу с градиентом (симуляция через множество сегментов)
    for angle in range(int(end_angle)):
        # Интерполяция цвета от teal к yellow
        progress = angle / 360.0
        
        # RGB значения для accent_teal (#3F888F) и accent_yellow (#E0C56B)
        teal_r, teal_g, teal_b = 63, 136, 143
        yellow_r, yellow_g, yellow_b = 224, 197, 107
        
        # Интерполированный цвет
        r = int(teal_r + (yellow_r - teal_r) * progress)
        g = int(teal_g + (yellow_g - teal_g) * progress)
        b = int(teal_b + (yellow_b - teal_b) * progress)
        
        segment_color = f"#{r:02x}{g:02x}{b:02x}"
        
        # Рисуем сегмент дуги (начинаем с верха - позиция "12 часов")
        start_angle = -90 + angle  # -90 чтобы начать сверху
        draw.arc([(center_x - radius, center_y - radius), 
                  (center_x + radius, center_y + radius)], 
                 start=start_angle, end=start_angle + 1, 
                 fill=segment_color, width=ring_width)
    
    # Центральный текст - численное значение (улучшенное позиционирование)
    value_text = f"{harmony_index:.2f}"
    value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
    value_width = value_bbox[2] - value_bbox[0]
    value_height = value_bbox[3] - value_bbox[1]
    # Размещаем чуть выше центра
    draw.text((center_x - value_width//2, center_y - value_height//2 - (20 * SCALE_FACTOR)), 
              value_text, fill=PALETTE['text_main'], font=value_font)
    
    # Подпись под числом (улучшенное позиционирование)
    if lang == 'ru':
        label_text = "Индекс Гармонии"
    else:
        label_text = "Architectural Harmony Index"
    
    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]
    # Размещаем ниже центра
    draw.text((center_x - label_width//2, center_y + (40 * SCALE_FACTOR)), 
              label_text, fill=PALETTE['text_light'], font=label_font)
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"Круговая диаграмма гармонии сохранена: {output_path}")


def create_subindex_chart(function_index: float, beauty_index: float, output_path: str, lang: str = 'ru'):
    """
    Создает горизонтальную столбчатую диаграмму для сравнения Индекса Функции и Индекса Красоты.
    
    Args:
        function_index (float): Значение индекса функции от 0 до 1
        beauty_index (float): Значение индекса красоты от 0 до 1
        output_path (str): Путь для сохранения изображения
        lang (str): Язык ('ru' или 'en')
    """
    # Создаем холст с высоким разрешением
    img = Image.new('RGB', (600 * SCALE_FACTOR, 400 * SCALE_FACTOR), PALETTE['background'])
    draw = ImageDraw.Draw(img)
    
    # Загружаем шрифты с учетом языка и масштабирования
    title_font = _load_font(24 * SCALE_FACTOR, bold=True, lang=lang)
    label_font = _load_font(16 * SCALE_FACTOR, lang=lang)
    value_font = _load_font(14 * SCALE_FACTOR, bold=True, lang=lang)
    
    # Заголовок слева вверху (отступ с масштабированием)
    title_text = TRANSLATIONS[lang]['subindex_chart_title']
    draw.text((40 * SCALE_FACTOR, 40 * SCALE_FACTOR), title_text, fill=PALETTE['text_main'], font=title_font)
    
    # Параметры горизонтальных полос с масштабированием
    chart_left = 200 * SCALE_FACTOR
    chart_width = 300 * SCALE_FACTOR
    bar_height = 40 * SCALE_FACTOR
    
    # Позиции полос по Y с масштабированием
    func_y = 150 * SCALE_FACTOR
    beauty_y = 220 * SCALE_FACTOR
    
    # Рисуем вертикальные линии сетки
    for i in range(6):  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        x = chart_left + (i * chart_width // 5)
        draw.line([(x, 120 * SCALE_FACTOR), (x, 280 * SCALE_FACTOR)], fill=PALETTE['grid'], width=1 * SCALE_FACTOR)
    
    # Рисуем полосу функции (бирюзовая)
    func_width = int(function_index * chart_width)
    draw.rectangle([(chart_left, func_y), (chart_left + func_width, func_y + bar_height)], 
                  fill=PALETTE['accent_teal'])
    
    # Рисуем полосу красоты (оранжевая)
    beauty_width = int(beauty_index * chart_width)
    draw.rectangle([(chart_left, beauty_y), (chart_left + beauty_width, beauty_y + bar_height)], 
                  fill=PALETTE['accent_orange'])
    
    # Подписи слева от полос
    func_label = TRANSLATIONS[lang]['function_index']
    beauty_label = TRANSLATIONS[lang]['beauty_index']
    
    # Выравниваем подписи по правому краю относительно начала полос
    func_bbox = draw.textbbox((0, 0), func_label, font=label_font)
    func_label_width = func_bbox[2] - func_bbox[0]
    draw.text((chart_left - func_label_width - (20 * SCALE_FACTOR), func_y + (10 * SCALE_FACTOR)), func_label, fill=PALETTE['text_main'], font=label_font)
    
    beauty_bbox = draw.textbbox((0, 0), beauty_label, font=label_font)
    beauty_label_width = beauty_bbox[2] - beauty_bbox[0]
    draw.text((chart_left - beauty_label_width - (20 * SCALE_FACTOR), beauty_y + (10 * SCALE_FACTOR)), beauty_label, fill=PALETTE['text_main'], font=label_font)
    
    # Значения в конце полос
    func_value = f"{function_index:.2f}"
    beauty_value = f"{beauty_index:.2f}"
    
    draw.text((chart_left + func_width + (10 * SCALE_FACTOR), func_y + (12 * SCALE_FACTOR)), func_value, fill=PALETTE['text_main'], font=value_font)
    draw.text((chart_left + beauty_width + (10 * SCALE_FACTOR), beauty_y + (12 * SCALE_FACTOR)), beauty_value, fill=PALETTE['text_main'], font=value_font)
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"Диаграмма составляющих сохранена: {output_path}")

def create_details_view(details: dict, output_path: str, lang: str = 'ru'):
    """
    Создает изображение с детальными показателями функции и красоты.
    
    Args:
        details (dict): Словарь с показателями {'thermal_score': 0.8, 'optic_score': 0.7, 'color_score': 0.9, ...}
        output_path (str): Путь для сохранения изображения
        lang (str): Язык ('ru' или 'en')
    """
    # Создаем холст с высоким разрешением
    img = Image.new('RGB', (600 * SCALE_FACTOR, 500 * SCALE_FACTOR), PALETTE['background'])
    draw = ImageDraw.Draw(img)
    
    # Загружаем шрифты с учетом языка и масштабирования
    title_font = _load_font(24 * SCALE_FACTOR, bold=True, lang=lang)
    section_font = _load_font(18 * SCALE_FACTOR, bold=True, lang=lang)
    text_font = _load_font(14 * SCALE_FACTOR, lang=lang)
    
    # Заголовок слева вверху (отступ с масштабированием)
    title_text = TRANSLATIONS[lang]['details_view_title']
    draw.text((40 * SCALE_FACTOR, 40 * SCALE_FACTOR), title_text, fill=PALETTE['text_main'], font=title_font)
    
    # Левая колонка - Функция
    draw.text((60 * SCALE_FACTOR, 120 * SCALE_FACTOR), TRANSLATIONS[lang]['function'], fill=PALETTE['accent_teal'], font=section_font)
    
    # Термодинамика
    thermal_score = details.get('thermal_score', 0.0)
    thermo_text = f"{TRANSLATIONS[lang]['thermodynamics']}: {thermal_score:.2f}"
    draw.text((80 * SCALE_FACTOR, 160 * SCALE_FACTOR), thermo_text, fill=PALETTE['text_main'], font=text_font)
    
    # Стильный progress bar для термодинамики
    bar_width = 180 * SCALE_FACTOR
    bar_height = 12 * SCALE_FACTOR
    bar_x = 80 * SCALE_FACTOR
    bar_y = 180 * SCALE_FACTOR
    
    # Фоновая полоса
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                  fill=PALETTE['grid'])
    # Активная полоса
    active_width = int(thermal_score * bar_width)
    draw.rectangle([(bar_x, bar_y), (bar_x + active_width, bar_y + bar_height)], 
                  fill=PALETTE['accent_yellow'])
    
    # Оптика
    optic_score = details.get('optic_score', 0.0)
    optic_text = f"{TRANSLATIONS[lang]['optics']}: {optic_score:.2f}"
    draw.text((80 * SCALE_FACTOR, 220 * SCALE_FACTOR), optic_text, fill=PALETTE['text_main'], font=text_font)
    
    # Progress bar для оптики
    bar_y = 240 * SCALE_FACTOR
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                  fill=PALETTE['grid'])
    active_width = int(optic_score * bar_width)
    draw.rectangle([(bar_x, bar_y), (bar_x + active_width, bar_y + bar_height)], 
                  fill=PALETTE['accent_yellow'])
    
    # Правая колонка - Красота
    draw.text((340 * SCALE_FACTOR, 120 * SCALE_FACTOR), TRANSLATIONS[lang]['beauty'], fill=PALETTE['accent_orange'], font=section_font)
    
    # Цветовая гармония
    color_score = details.get('color_score', 0.0)
    color_text = f"{TRANSLATIONS[lang]['color_harmony']}: {color_score:.2f}"
    draw.text((360 * SCALE_FACTOR, 160 * SCALE_FACTOR), color_text, fill=PALETTE['text_main'], font=text_font)
    
    # Progress bar для цвета
    bar_x = 360 * SCALE_FACTOR
    bar_y = 180 * SCALE_FACTOR
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                  fill=PALETTE['grid'])
    active_width = int(color_score * bar_width)
    draw.rectangle([(bar_x, bar_y), (bar_x + active_width, bar_y + bar_height)], 
                  fill=PALETTE['accent_yellow'])
    
    # Стилевое единство
    style_score = details.get('style_score', 0.0)
    style_text = f"{TRANSLATIONS[lang]['style_unity']}: {style_score:.2f}"
    draw.text((360 * SCALE_FACTOR, 220 * SCALE_FACTOR), style_text, fill=PALETTE['text_main'], font=text_font)
    
    # Progress bar для стиля
    bar_y = 240 * SCALE_FACTOR
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                  fill=PALETTE['grid'])
    active_width = int(style_score * bar_width)
    draw.rectangle([(bar_x, bar_y), (bar_x + active_width, bar_y + bar_height)], 
                  fill=PALETTE['accent_yellow'])
    
    # Пространство
    space_score = details.get('space_score', 0.0)
    space_text = f"{TRANSLATIONS[lang]['space']}: {space_score:.2f}"
    draw.text((360 * SCALE_FACTOR, 280 * SCALE_FACTOR), space_text, fill=PALETTE['text_main'], font=text_font)
    
    # Progress bar для пространства
    bar_y = 300 * SCALE_FACTOR
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                  fill=PALETTE['grid'])
    active_width = int(space_score * bar_width)
    draw.rectangle([(bar_x, bar_y), (bar_x + active_width, bar_y + bar_height)], 
                  fill=PALETTE['accent_yellow'])
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"Детальные показатели сохранены: {output_path}")


def create_before_after_chart(before_scores: dict, after_scores: dict, output_path: str, lang: str = 'ru'):
    """
    Создает диаграмму сравнения индексов "До" и "После".
    
    Args:
        before_scores (dict): Словарь с индексами до улучшений {'harmony': 0.5, 'function': 0.6, 'beauty': 0.4}
        after_scores (dict): Словарь с индексами после улучшений {'harmony': 0.8, 'function': 0.9, 'beauty': 0.7}
        output_path (str): Путь для сохранения изображения
        lang (str): Язык ('ru' или 'en')
    """
    # Создаем холст с высоким разрешением
    img = Image.new('RGB', (800 * SCALE_FACTOR, 600 * SCALE_FACTOR), PALETTE['background'])
    draw = ImageDraw.Draw(img)
    
    # Загружаем шрифты с учетом языка и масштабирования
    title_font = _load_font(24 * SCALE_FACTOR, bold=True, lang=lang)
    label_font = _load_font(14 * SCALE_FACTOR, lang=lang)
    value_font = _load_font(12 * SCALE_FACTOR, bold=True, lang=lang)
    legend_font = _load_font(12 * SCALE_FACTOR, lang=lang)
    
    # Заголовок слева вверху (отступ с масштабированием)
    title_text = TRANSLATIONS[lang]['before_after_title']
    draw.text((40 * SCALE_FACTOR, 40 * SCALE_FACTOR), title_text, fill=PALETTE['text_main'], font=title_font)
    
    
    # Легенда под заголовком (слева)
    legend_y = 80 * SCALE_FACTOR
    # "До" - светло-серый квадрат
    draw.rectangle([(40 * SCALE_FACTOR, legend_y), (55 * SCALE_FACTOR, legend_y + (12 * SCALE_FACTOR))], fill=PALETTE['grid'])
    draw.text((65 * SCALE_FACTOR, legend_y), TRANSLATIONS[lang]['before'], fill=PALETTE['text_main'], font=legend_font)
    
    # "После" - бирюзовый квадрат
    draw.rectangle([(150 * SCALE_FACTOR, legend_y), (165 * SCALE_FACTOR, legend_y + (12 * SCALE_FACTOR))], fill=PALETTE['accent_teal'])
    draw.text((175 * SCALE_FACTOR, legend_y), TRANSLATIONS[lang]['after'], fill=PALETTE['text_main'], font=legend_font)
    
    # Параметры диаграммы с масштабированием
    chart_height = 300 * SCALE_FACTOR
    chart_bottom = 450 * SCALE_FACTOR
    bar_width = 50 * SCALE_FACTOR
    
    # Позиции групп столбцов с масштабированием
    harmony_x = 200 * SCALE_FACTOR
    function_x = 400 * SCALE_FACTOR
    beauty_x = 600 * SCALE_FACTOR
    
    # Рисуем горизонтальные линии сетки
    for i in range(6):  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        y = chart_bottom - (i * chart_height // 5)
        draw.line([(150 * SCALE_FACTOR, y), (650 * SCALE_FACTOR, y)], fill=PALETTE['grid'], width=1 * SCALE_FACTOR)
    
    # Данные для диаграммы
    categories = [TRANSLATIONS[lang]['harmony'], TRANSLATIONS[lang]['function'], TRANSLATIONS[lang]['beauty']]
    before_values = [before_scores.get('harmony', 0), before_scores.get('function', 0), before_scores.get('beauty', 0)]
    after_values = [after_scores.get('harmony', 0), after_scores.get('function', 0), after_scores.get('beauty', 0)]
    positions = [harmony_x, function_x, beauty_x]
    
    # Рисуем столбцы для каждой категории
    for i, (category, before_val, after_val, x_pos) in enumerate(zip(categories, before_values, after_values, positions)):
        # Высоты столбцов
        before_height = int(before_val * chart_height)
        after_height = int(after_val * chart_height)
        
        # Столбец "До" (светло-серый, левее)
        before_x = x_pos - bar_width//2 - (15 * SCALE_FACTOR)
        before_top = chart_bottom - before_height
        draw.rectangle([(before_x - bar_width//2, before_top), 
                       (before_x + bar_width//2, chart_bottom)], 
                      fill=PALETTE['grid'])
        
        # Столбец "После" (бирюзовый, правее)
        after_x = x_pos + bar_width//2 + (15 * SCALE_FACTOR)
        after_top = chart_bottom - after_height
        draw.rectangle([(after_x - bar_width//2, after_top), 
                       (after_x + bar_width//2, chart_bottom)], 
                      fill=PALETTE['accent_teal'])
        
        # Значения над столбцами
        before_text = f"{before_val:.2f}"
        after_text = f"{after_val:.2f}"
        
        before_bbox = draw.textbbox((0, 0), before_text, font=value_font)
        before_width = before_bbox[2] - before_bbox[0]
        draw.text((before_x - before_width//2, before_top - (25 * SCALE_FACTOR)), before_text, fill=PALETTE['text_main'], font=value_font)
        
        after_bbox = draw.textbbox((0, 0), after_text, font=value_font)
        after_width = after_bbox[2] - after_bbox[0]
        draw.text((after_x - after_width//2, after_top - (25 * SCALE_FACTOR)), after_text, fill=PALETTE['text_main'], font=value_font)
        
        # Подписи категорий
        category_bbox = draw.textbbox((0, 0), category, font=label_font)
        category_width = category_bbox[2] - category_bbox[0]
        draw.text((x_pos - category_width//2, 480 * SCALE_FACTOR), category, fill=PALETTE['text_main'], font=label_font)
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"Диаграмма сравнения до/после сохранена: {output_path}")
