import pandas as pd
import numpy as np
import joblib
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

def transformar(datos_entrada):
    
    # 1. Dataframe de los datos de entrada
    df_datos = pd.DataFrame(datos_entrada, index=[0])
    
    # Unir el título con la trama para que sea un mismo texto para el encoder
    df_datos['title_plot'] = df_datos['title'] + ' - ' + df_datos['plot']
    
    # Eliminar columnas que se unieron
    df_datos.drop(columns=['title','plot'], inplace=True)
    
    #Scaler
    scaler = joblib.load(os.path.dirname(__file__) + '/scaler.pkl')
    
    #Embedding
    module_url = os.path.dirname(__file__) + '/universal-sentence-encoder_4/'
    
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.load(module_url)
        embedded_text = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    
    session = tf.Session(graph=g)
    session.run(init_op)
    
    result = session.run(embedded_text, feed_dict={text_input: df_datos['title_plot']})
    
    x_test_embed = pd.DataFrame(result)
    x_test_embed.index = df_datos.index
    
    # Concateno los embedding realizados con la tabla original para traer el año
    df_datos_2 = pd.concat([df_datos, x_test_embed], axis=1)
    df_datos_2.drop(columns=['title_plot'], inplace=True)
    # Una vez tengo el DF con el año y los embedding, se escala todo (el año es el que lo requiere)
    df_datos_2 = scaler.transform(df_datos_2)
    
    # Modelo
    modelo_cargado = load_model(os.path.dirname(__file__) + '/modelo_red_neuronal_h5.h5')
    
    # Predicciones
    y_pred_genres = modelo_cargado.predict(df_datos_2)[0]

    # dar formato a predicciones
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    res = pd.DataFrame(y_pred_genres).T
    res.columns = cols
    
    return res.to_dict()
    
    