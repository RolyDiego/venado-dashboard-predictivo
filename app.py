import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, time
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Predictivo Venado",
    layout="wide"
)

st.title("Dashboard Predictivo de Rutas - Industrias Venado")

# Cargar modelos
travel_model = joblib.load("models/modelo_tiempo_recorrido.pkl")
task_model = joblib.load("models/modelo_duracion_microtareas.pkl")
task_columns = joblib.load("models/task_model_columns.pkl")

# Cargar datasets base
df_travel = pd.read_excel(
    "data/datasets_predictivos_venado.xlsx",
    sheet_name="travel_time_dataset"
)

df_tasks = pd.read_excel(
    "data/datasets_predictivos_venado.xlsx",
    sheet_name="task_duration_dataset"
)

# Sidebar
st.sidebar.header("Filtros de predicción")

fecha = st.sidebar.date_input(
    "Seleccionar fecha",
    value=datetime.today()
)

reponedor = st.sidebar.selectbox(
    "Seleccionar reponedor",
    ["REPONEDOR 1", "REPONEDOR 2", "REPONEDOR 3"]
)

hora_inicio = st.sidebar.time_input(
    "Hora inicio jornada",
    value=time(8, 0)
)

cantidad_pdvs = st.sidebar.slider(
    "Cantidad de PDVs a visitar",
    min_value=5,
    max_value=30,
    value=8
)

zona = st.sidebar.selectbox(
    "Tipo de zona",
    ["Baja congestión", "Media congestión", "Alta congestión"]
)

boton = st.sidebar.button("Generar predicción")

zone_map = {
    "Baja congestión": 0,
    "Media congestión": 1,
    "Alta congestión": 2
}

day_of_week = fecha.weekday()
zone_type = zone_map[zona]

def predecir_tarea(customer_type, task_type, merchandiser_experience, store_size, product_quantity):
    row = pd.DataFrame([{
        "merchandiser_experience": merchandiser_experience,
        "store_size": store_size,
        "product_quantity": product_quantity,
        f"customer_type_{customer_type}": 1,
        f"task_type_{task_type}": 1
    }])

    for col in task_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[task_columns]

    return task_model.predict(row)[0]

if boton:
    np.random.seed(42)

    resultados = []

    hora_actual = datetime.combine(fecha, hora_inicio)

    for i in range(1, cantidad_pdvs + 1):
        distance_km = np.random.uniform(0.5, 8.0)

        travel_input = pd.DataFrame([{
            "distance_km": distance_km,
            "hour": hora_actual.hour,
            "day_of_week": day_of_week,
            "zone_type": zone_type
        }])

        tiempo_recorrido = travel_model.predict(travel_input)[0]

        customer_type = np.random.choice(
            ["MAYORISTA", "MINORISTA", "PARETO"],
            p=[0.25, 0.5, 0.25]
        )

        store_size = np.random.choice([0, 1, 2])
        product_quantity = np.random.randint(20, 200)
        experience = np.random.choice([0, 1, 2])

        if customer_type == "MINORISTA":
            tareas_base = ["REPOSICION", "POP"]
        elif customer_type == "MAYORISTA":
            tareas_base = ["STOCK", "REPOSICION", "POP"]
        else:
            tareas_base = ["STOCK", "REPOSICION", "EXHIBICION", "POP"]

        tiempo_tareas = 0

        for tarea in tareas_base:
            tiempo_tareas += predecir_tarea(
                customer_type=customer_type,
                task_type=tarea,
                merchandiser_experience=experience,
                store_size=store_size,
                product_quantity=product_quantity
            )

        total_pdv = tiempo_recorrido + tiempo_tareas

        llegada = hora_actual + timedelta(minutes=float(tiempo_recorrido))
        salida = llegada + timedelta(minutes=float(tiempo_tareas))

        resultados.append({
            "PDV": f"GV{i:03d}",
            "Tipo cliente": customer_type,
            "Distancia km": round(distance_km, 2),
            "Tiempo recorrido": round(tiempo_recorrido, 2),
            "Tiempo tareas": round(tiempo_tareas, 2),
            "Tiempo total PDV": round(total_pdv, 2),
            "Hora llegada": llegada.strftime("%H:%M"),
            "Hora salida": salida.strftime("%H:%M")
        })

        hora_actual = salida

    df_result = pd.DataFrame(resultados)

    total_recorrido = df_result["Tiempo recorrido"].sum()
    total_tareas = df_result["Tiempo tareas"].sum()
    total_jornada = df_result["Tiempo total PDV"].sum()
    hora_fin = hora_actual.strftime("%H:%M")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PDVs estimados", cantidad_pdvs)
    col2.metric("Tiempo recorrido", f"{round(total_recorrido, 1)} min")
    col3.metric("Tiempo tareas", f"{round(total_tareas, 1)} min")
    col4.metric("Hora fin estimada", hora_fin)

    st.subheader("Detalle de predicción por PDV")
    st.dataframe(df_result, use_container_width=True)

    st.subheader("Tiempo total por PDV")

    fig1 = px.bar(
        df_result,
        x="PDV",
        y="Tiempo total PDV",
        color="Tipo cliente",
        title="Tiempo estimado total por punto de venta"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Distribución del tiempo")

    df_pie = pd.DataFrame({
        "Concepto": ["Recorrido", "Microtareas"],
        "Minutos": [total_recorrido, total_tareas]
    })

    fig2 = px.pie(
        df_pie,
        values="Minutos",
        names="Concepto",
        title="Distribución entre traslado y trabajo operativo"
    )

    st.plotly_chart(fig2, use_container_width=True)

    if total_jornada > 480:
        st.error("⚠️ Ruta sobrecargada: supera una jornada laboral de 8 horas")
    else:
        st.success("✅ Ruta dentro de una jornada laboral estimada")

else:
    st.info("Selecciona los filtros y presiona 'Generar predicción'.")