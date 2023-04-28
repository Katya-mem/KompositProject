from flask import Flask, render_template, request
import pandas as pd
import pickle

MODELS_PATH = 'flask_app/models/'


# Извлечение данных
def get_data_from_form(features, params):
    parameter_names = features.keys()
    data = dict.fromkeys(parameter_names, None)
    error = ''
    # Преобразование строк в числа
    for param_name, param_value in params.items():
        if param_value.strip() != '':
            try:
                data[param_name] = float(param_value)
            except:
                error += f'{features[param_name]} - некорректное значение "{param_value}"\n'
    # Соотношение матрица-наполнитель
    if 'par1' in data and data['par1'] is not None:
        if data['par1'] < 0:
            error += f'{features["par1"]} - значение вне корректного диапазона\n'
    # Плотность, кг/м3
    if 'par2' in data and data['par2'] is not None:
        if data['par2'] < 0:
            error += f'{features["par2"]} - значение вне корректного диапазона\n'
    # Модуль упругости, ГПа
    if 'par3' in data and data['par3'] is not None:
        if data['par3'] < 0:
            error += f'{features["par3"]} - значение вне корректного диапазона\n'
    # 'Количество отвердителя, м.%
    if 'par4' in data and data['par4'] is not None:
        if data['par4'] < 0:
            error += f'{features["par4"]} - значение вне корректного диапазона\n'
    # Содержание эпоксидных групп,%_2
    if 'par5' in data and data['par5'] is not None:
        if data['par5'] < 0:
            error += f'{features["par5"]} - значение вне корректного диапазона\n'
    # Температура вспышки, С_2
    if 'par6' in data and data['par6'] is not None:
        if data['par6'] < 0:
            error += f'{features["par6"]} - значение вне корректного диапазона\n'
    # Поверхностная плотность, г/м2
    if 'par7' in data and data['par7'] is not None:
        if data['par7'] < 0:
            error += f'{features["par7"]} - значение вне корректного диапазона\n'
    # Модуль упругости при растяжении, ГПа
    if 'par8' in data and data['par8'] is not None:
        if data['par8'] < 0:
            error += f'{features["par8"]} - значение вне корректного диапазона\n'
    # Прочность при растяжении, МПа
    if 'par9' in data and data['par9'] is not None:
        if data['par9'] < 0:
            error += f'{features["par9"]} - значение вне корректного диапазона\n'
    # Потребление смолы, г/м2
    if 'par10' in data and data['par10'] is not None:
        if data['par10'] < 0:
            error += f'{features["par10"]} - значение вне корректного диапазона\n'
    # Угол нашивки, град
    if 'par11' in data and data['par11'] is not None:
        if data['par11'] != 0.0 and data['par11'] != 90.0:
            error += f'{features["par11"]} - значение вне корректного диапазона\n'
    # Плотность нашивки
    if 'par12' in data and data['par12'] is not None:
        if data['par12'] < 0:
            error += f'{features["par12"]} - значение вне корректного диапазона\n'
    # Шаг нашивки
    if 'par13' in data and data['par13'] is not None:
        if data['par13'] < 0:
            error += f'{features["par13"]} - значение вне корректного диапазона\n'
    # Проверка отсутствующих значений
    if None in data.values():
        error += f'Некоторые значения отсутствуют!\n'
    # Замена сокращенных имен признаков на полные
    data_clean = dict(zip(features.values(), data.values()))
    return data_clean, error


# Загрузка объекта
def load_pickle_obj(filename):
    file = open(MODELS_PATH + filename, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


app = Flask(__name__)


@app.route('/model_1_2/', methods=['post', 'get'])
def model_1_2_page():
    # Вводимые параметры
    features = {
        'par1': 'Соотношение матрица-наполнитель',
        'par2': 'Плотность, кг/м3',
        'par3': 'модуль упругости, ГПа',
        'par4': 'Количество отвердителя, м.%',
        'par5': 'Содержание эпоксидных групп,%_2',
        'par6': 'Температура вспышки, С_2',
        'par7': 'Поверхностная плотность, г/м2',
        'par10': 'Потребление смолы, г/м2',
        'par11': 'Угол нашивки, град',
        'par12': 'Шаг нашивки',
        'par13': 'Плотность нашивки'

    }
    params = dict(zip(features.keys(), ['2.771331', '2030.0',
                                        '753.0', '111.86',
                                        '22.267857', '284.615385',
                                        '210.0', '220.0',
                                        '0.0', '5.0', '57.0']))
    error = ''
    x = pd.DataFrame()
    par8 = ''
    par9 = ''
    # Получение данных
    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(features, params)
        if error == '':
            # Предсказание
            x = pd.DataFrame(data, index=[0])
            # Модуля упругости при растяжении
            preprocessor1 = load_pickle_obj('preprocessor1')
            model1 = load_pickle_obj('model1_best')
            x1 = preprocessor1.transform(x)
            y1 = model1.predict(x1)
            par8 = y1[0]
            # Прочности при растяжении
            preprocessor2 = load_pickle_obj('preprocessor2')
            model2 = load_pickle_obj('model2_best')
            x2 = preprocessor2.transform(x)
            y2 = model2.predict(x2)
            par9 = y2[0]
    return render_template('model_1_2.html', params=params, error=error, inputs=x.to_html(), par8=par8, par9=par9)


@app.route('/model_neuro/', methods=['post', 'get'])
def model_neuro_page():
    features = {
        'par2': 'Плотность, кг/м3',
        'par3': 'модуль упругости, ГПа',
        'par4': 'Количество отвердителя, м.%',
        'par5': 'Содержание эпоксидных групп,%_2',
        'par6': 'Температура вспышки, С_2',
        'par7': 'Поверхностная плотность, г/м2',
        'par8': 'Модуль упругости при растяжении, ГПа',
        'par9': 'Прочность при растяжении, МПа',
        'par10': 'Потребление смолы, г/м2',
        'par11': 'Угол нашивки, град',
        'par12': 'Шаг нашивки',
        'par13': 'Плотность нашивки'
    }
    params = dict(zip(features.keys(), ['2030.0', '753.0',
                                        '111.86', '22.267857',
                                        '284.615385', '210.0',
                                        '70.0', '3000.0',
                                        '220.0', '0.0',
                                        '5.0', '57.0']))
    error = ''
    x = pd.DataFrame()
    par1 = ''
    # Получение данных
    if request.method == 'POST':
        params = request.form.to_dict()
        data, error = get_data_from_form(features, params)
        if error == '':
            # Предсказание
            x = pd.DataFrame(data, index=[0])
            # Соотношение матрица-наполнитель
            preprocessor3 = load_pickle_obj('preprocessor3')
            model3 = load_pickle_obj('model3_neuro')
            x3 = preprocessor3.transform(x)
            y3 = model3.predict(x3)
            par1 = y3[0]
    # Отображение результата
    return render_template('model_neuro.html', params=params, error=error, inputs=x.to_html(), par1=par1)


@app.route('/')
def main_page():
    return render_template('main.html')


@app.route('/url_map/')
def url_map():
    return str(app.url_map)


if __name__ == '__main__':
    app.run()
