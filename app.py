import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split # Importar train_test_split
from sklearn.metrics import mean_absolute_error, r2_score # Para evaluar el modelo
import os # Importar el módulo os

# --- Carga de datos ---
# Asegúrate de que el archivo 'crabs.csv' esté en una carpeta 'data_sets'
# en el mismo directorio que este script.
try:
    df_crabs = pd.read_csv('data_sets/crabs.csv')
except FileNotFoundError:
    st.error("Error: El archivo 'crabs.csv' no se encontró.")
    st.error("Asegúrate de que esté en una carpeta 'data_sets' en el mismo directorio que este script.")
    st.stop() # Detener la ejecución de la app si el archivo no se encuentra

# --- Funciones de Regresión y Ploteo ---

# Función para generar predicciones de peso para diferentes partes del cangrejo
def generate_weight_predictions(input_weight, dataframe):
    """
    Realiza una regresión lineal para predecir los pesos de la Carne (Shucked),
    Vísceras (Viscera) y Caparazón (Shell) basándose en el 'new_weight' (peso total)
    del cangrejo, usando una división de datos para entrenamiento y prueba.
    También calcula y devuelve las métricas del modelo para los conjuntos de
    entrenamiento y prueba.

    Args:
        input_weight (float): El peso total de entrada del cangrejo.
        dataframe (pd.DataFrame): DataFrame con los datos de los cangrejos.

    Returns:
        tuple: Una tupla que contiene:
            - list: Pesos predichos para [Peso de Carne, Peso de Vísceras, Peso de Caparazón].
            - dict: Métricas del modelo para cada columna objetivo, incluyendo
                    entrenamiento y prueba.
                    Formato: {col_objetivo: {'entrenamiento': {'MAE': val, 'R2': val},
                                            'prueba': {'MAE': val, 'R2': val}}}
    """
    # Definir la variable independiente (X) y las variables dependientes (Y)
    X = dataframe["new_weight"].values.reshape((-1, 1))
    
    # Lista para almacenar los valores predichos
    predicted_values = []
    
    # Diccionario para almacenar métricas de evaluación para entrenamiento y prueba
    all_model_metrics = {}

    # Columnas de peso a predecir
    target_columns = ["Shucked Weight", "Viscera Weight", "Shell Weight"]

    for target_col in target_columns:
        y = dataframe[target_col]

        # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
        # Usamos random_state para asegurar la reproducibilidad de la división
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inicializar y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Realizar predicciones en el conjunto de entrenamiento y prueba para evaluación
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas para el conjunto de ENTRENAMIENTO
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        # Calcular métricas para el conjunto de PRUEBA
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Almacenar métricas
        all_model_metrics[target_col] = {
            "entrenamiento": {"Error Absoluto Medio (MAE)": mae_train, "Coeficiente de Determinación (R²)": r2_train},
            "prueba": {"Error Absoluto Medio (MAE)": mae_test, "Coeficiente de Determinación (R²)": r2_test}
        }

        # Hacer la predicción para el input_weight (siempre se hace sobre el modelo entrenado)
        prediction = model.predict([[input_weight]])[0]
        predicted_values.append(max(0, prediction)) # Asegurar que los valores no sean negativos

    return predicted_values, all_model_metrics

# Función para generar el gráfico de barras
def plot_bar_chart(values):
    """
    Genera un gráfico de barras comparando los pesos predichos.

    Args:
        values (list): Lista de pesos predichos [Carne, Vísceras, Caparazón].

    Returns:
        matplotlib.figure.Figure: La figura del gráfico de barras.
    """
    fig, ax = plt.subplots(figsize=(8, 6)) # Tamaño de figura reducido

    # Ajustar tamaños de elementos en el gráfico para una mejor visualización en Streamlit
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=14)

    # Usando una paleta de colores normal y variada
    sns.set_style("whitegrid")
    sns.barplot(x=["Shucked Weight", "Viscera Weight", "Shell Weight"], y=values, palette="viridis", ax=ax)

    ax.set_ylim(0, max(values) * 1.2 if values else 1) # Ajustar límite Y para mostrar etiquetas
    ax.set_title('Comparación de Pesos', fontsize=20)
    ax.set_ylabel('Peso (gramos)')

    # Añadir etiquetas de valor en las barras
    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.05 if values else 0.01), f"{round(v, 2)}", ha='center', va='bottom', fontsize=12)

    return fig

# Función para generar el gráfico circular (pie chart)
def generate_pie_chart(values):
    """
    Genera un gráfico circular mostrando la distribución porcentual de los pesos.

    Args:
        values (list): Lista de pesos predichos [Carne, Vísceras, Caparazón].

    Returns:
        matplotlib.figure.Figure: La figura del gráfico circular.
    """
    fig, ax = plt.subplots(figsize=(7, 7)) # Tamaño de figura reducido

    # Ajustar tamaños de elementos en el gráfico
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=14)

    # Usando una paleta de colores personalizada, normal y variada
    colors = ['#87CEFA', '#C82A54', '#00FA9A'] # Azul cielo suave, Rosa rojizo suave, Amarillo ocre suave (Paleta 1)

    categories = ["Shucked Weight", "Viscera Weight", "Shell Weight"]

    # Asegurar que la suma de los valores sea mayor que cero para evitar errores en el gráfico circular
    if sum(values) > 0:
        wedges, texts, autotexts = ax.pie(values,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          wedgeprops={'edgecolor': 'black'})

        # Añadir leyenda fuera del gráfico para mayor claridad
        ax.legend(wedges, categories,
                  title="Categorías",
                  loc="center left",
                  bbox_to_anchor=(1.05, 0, 0.3, 1))
    else:
        # Si todos los valores son cero, mostrar un gráfico circular vacío o un mensaje
        ax.text(0.5, 0.5, "No hay datos para mostrar",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])


    ax.set_title('Distribución de Categorías', fontsize=20)
    plt.setp(autotexts, size=12, weight="bold") # Ajustar tamaño y negrita de los porcentajes

    return fig

# --- Función principal de la aplicación Streamlit ---
def main():
    st.set_page_config(layout="wide", page_title="Análisis de Cangrejos")

    # Contenido de la barra lateral
    with st.sidebar:
        # Usar Markdown para controlar el tamaño de fuente del encabezado
        st.markdown("## Inserta el peso del cangrejo") # Tamaño de fuente aumentado

        # Entrada numérica para el peso del cangrejo
        inp_value = st.number_input(
            "Ingrese un valor de peso (mayor que 4, menor o igual a 100):",
            min_value=4.0, max_value=100.0, value=5.0, step=0.1
        )

        # Botón para activar la predicción
        if st.button("Calcular Predicciones"):
            pass


        # Imagen del cangrejo en la barra lateral, debajo del botón
        # Construir la ruta absoluta a la imagen para mayor robustez
        current_dir = os.path.dirname(__file__)
        img_path_relative = os.path.join(current_dir, 'img', 'crab1.png')
        
        try:
            st.image(img_path_relative, caption="", use_container_width=True)
        except FileNotFoundError:
            st.warning("La imagen 'crab1.png' no se encontró en la ruta esperada ('img/').")
            # CÓDIGO MODIFICADO AQUÍ: cambiamos use_column_width a use_container_width
            st.image("https://placehold.co/400x300/cccccc/000000?text=Imagen+Cangrejo", caption="Placeholder de imagen", use_container_width=True)


    # Contenido principal de la aplicación
    # Título principal de la aplicación, ahora centrado sobre los gráficos
    st.title("Análisis de los Cangrejos 🦀") # Añadido emoji de cangrejo

    if inp_value <= 100 and inp_value > 4:
        # Generar predicciones y obtener métricas
        values, all_model_metrics = generate_weight_predictions(inp_value, df_crabs)

        st.subheader(f"Predicciones para {round(inp_value, 2)} gramos")

        # Columnas para los gráficos (Comparación de Pesos y Distribución por Categorías)
        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            fig_bar = plot_bar_chart(values)
            st.pyplot(fig_bar)

        with col_chart2:
            fig_pie = generate_pie_chart(values)
            st.pyplot(fig_pie)
        
        # --- Sección de Resumen de Producción ---
        st.subheader("Resumen de la Producción")

        # ELIMINADO: Bloque CSS inyectado que causaba advertencia
        # st.markdown("""
        # <style type="text/css">
        # #T_9c7bf_row0_col0, #T_9c7bf_row0_col1, #T_9c7bf_row1_col0, #T_9c7bf_row1_col1, #T_9c7bf_row2_col0, #T_9c7bf_row2_col1 { width: 300px; text-align: left;}
        # </style>
        # """, unsafe_allow_html=True)

        # Calcular valores de resumen
        total_meat_crab = values[0]
        utility_kitchen = values[0] + values[2] # Peso de Carne + Peso de Caparazón
        waste = values[1] # Peso de Vísceras

        # Preparar los datos para la tabla de resumen
        summary_data = {
            "Categoría": ["Carne producida 🥩", "Material de aprovechamiento 🍽️", "Desperdicio 🗑️"],
            "Peso (gramos)": [
                round(total_meat_crab, 2),
                round(utility_kitchen, 2),
                round(waste, 2)
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # CÓDIGO MODIFICADO AQUÍ: añadimos use_container_width=True a st.dataframe
        st.dataframe(df_summary.style.format(subset=["Peso (gramos)"], formatter="{:.2f}"), hide_index=True, use_container_width=True)

        # --- Sección de Métricas del Modelo en Tabla ---
        st.subheader("Métricas de Evaluación del Modelo (Entrenamiento y Prueba)")

        # Preparar los datos para la tabla Markdown
        metric_data = {
            "Métrica": [],
            "Shucked Weight (Entrenamiento)": [],
            "Shucked Weight (Prueba)": [],
            "Viscera Weight (Entrenamiento)": [],
            "Viscera Weight (Prueba)": [],
            "Shell Weight (Entrenamiento)": [],
            "Shell Weight (Prueba)": []
        }

        # Iterar sobre las métricas y poblar el diccionario
        for metric_name in ["Error Absoluto Medio (MAE)", "Coeficiente de Determinación (R²)"]:
            metric_data["Métrica"].append(metric_name)
            for target_col in ["Shucked Weight", "Viscera Weight", "Shell Weight"]:
                # Acceder a los valores de las métricas de forma segura
                metric_data[f"{target_col} (Entrenamiento)"].append(
                    all_model_metrics.get(target_col, {}).get("entrenamiento", {}).get(metric_name, np.nan)
                )
                metric_data[f"{target_col} (Prueba)"].append(
                    all_model_metrics.get(target_col, {}).get("prueba", {}).get(metric_name, np.nan)
                )

        # Crear un DataFrame de pandas
        df_metrics = pd.DataFrame(metric_data)

        # Identificar las columnas numéricas para el formato
        numeric_cols = [col for col in df_metrics.columns if col != "Métrica"]

        # CÓDIGO MODIFICADO AQUÍ: añadimos use_container_width=True a st.dataframe
        st.dataframe(df_metrics.style.format(subset=numeric_cols, formatter="{:.2f}"), hide_index=True, use_container_width=True)

    elif inp_value > 100:
        st.info("¡Los cangrejos no pesan eso! Por favor, ingresa un valor de peso realista.")
    else:
        st.warning("Por favor, ingrese un valor mayor que 4 para el peso del cangrejo.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
