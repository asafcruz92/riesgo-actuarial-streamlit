import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATASET_PATH = "dataset_con_clusters.csv"


st.set_page_config(
    page_title="Prediccion de riesgo actuarial",
    layout="wide",
)


@st.cache_data
def cargar_datos():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def entrenar_modelo(datos):
    variables_numericas = ["age", "bmi", "children", "charges"]
    variables_categoricas = ["sex", "smoker", "region"]

    X = datos[variables_numericas + variables_categoricas]
    y = datos["riesgo_actuarial"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), variables_numericas),
            ("categoricas", OneHotEncoder(drop="first", handle_unknown="ignore"), variables_categoricas),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            (
                "clasificador",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="sag",
                ),
            ),
        ]
    )

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)

    modelo.fit(X, y)

    return modelo, accuracy

def generar_recomendacion(riesgo, cliente):
    recomendaciones = []

    if riesgo == "Riesgo bajo":
        recomendaciones.append("Mantener controles preventivos regulares para conservar el perfil de bajo riesgo.")
        recomendaciones.append("Promover habitos saludables, como actividad fisica, alimentacion balanceada y seguimiento medico basico.")
        recomendaciones.append("Ofrecer beneficios de fidelizacion o educacion preventiva, ya que el costo esperado es menor.")
    elif riesgo == "Riesgo medio":
        recomendaciones.append("Dar seguimiento preventivo con chequeos medicos periodicos para evitar que el riesgo aumente.")
        recomendaciones.append("Revisar factores como edad, IMC y cargos medicos estimados para ajustar el monitoreo del cliente.")
        recomendaciones.append("Considerar programas de prevencion y acompanamiento, especialmente si existen factores modificables.")
    else:
        recomendaciones.append("Realizar una evaluacion actuarial mas cuidadosa por el mayor nivel de riesgo estimado.")
        recomendaciones.append("Promover acciones preventivas relacionadas con tabaquismo, control de peso y seguimiento medico.")
        recomendaciones.append("Analizar el costo esperado con mayor detalle antes de tomar decisiones de tarifacion.")

    if cliente["smoker"] == "yes":
        recomendaciones.append("Como el cliente es fumador, se recomienda incluir orientacion para reducir o abandonar el consumo de tabaco.")

    if cliente["bmi"] >= 30:
        recomendaciones.append("El IMC se encuentra en un rango alto, por lo que conviene sugerir control de peso y evaluacion medica.")

    texto = "\n".join([f"{i + 1}. {recomendacion}" for i, recomendacion in enumerate(recomendaciones[:4])])
    return texto

datos = cargar_datos()
modelo, accuracy = entrenar_modelo(datos)

st.title("PREDICCIÓN DE RIESGO ACTUARIAL - Asaf Cruz - PCAF-03")
st.caption("Regresion Logistica para clasificar riesgo actuarial")

with st.container(border=True):
    st.subheader("Formulario de ingreso de datos")

    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        fumador = st.selectbox("Fumador", ["no", "yes"])

    with col2:
        sexo = st.selectbox("Sexo", ["female", "male"])
        hijos = st.number_input("Hijos", min_value=0, max_value=10, value=1, step=1)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    cargos = st.number_input(
        "Cargos medicos estimados",
        min_value=0.0,
        max_value=100000.0,
        value=12000.0,
        step=500.0,
    )

    evaluar = st.button("Evaluar", type="primary")

cliente = pd.DataFrame(
    {
        "age": [edad],
        "sex": [sexo],
        "bmi": [bmi],
        "children": [hijos],
        "smoker": [fumador],
        "region": [region],
        "charges": [cargos],
    }
)

if evaluar:
    riesgo = modelo.predict(cliente)[0]
    probabilidades = modelo.predict_proba(cliente)[0]
    clases = modelo.classes_
    confianza = probabilidades.max()

    colores = {
        "Riesgo bajo": "green",
        "Riesgo medio": "orange",
        "Riesgo alto": "red",
    }

    st.markdown(
        f"## Riesgo actuarial: :{colores.get(riesgo, 'blue')}[{riesgo.replace('Riesgo ', '').capitalize()}]"
    )
    st.write(f"Confianza aproximada del modelo: **{confianza:.2%}**")

    st.subheader("Probabilidades por nivel de riesgo")
    tabla_probabilidades = pd.DataFrame(
        {
            "riesgo_actuarial": clases,
            "probabilidad": probabilidades,
        }
    ).sort_values("probabilidad", ascending=False)
    st.dataframe(tabla_probabilidades, use_container_width=True, hide_index=True)
        st.subheader("Recomendaciones automaticas")
    recomendacion = generar_recomendacion(riesgo, cliente.iloc[0].to_dict())
    st.markdown(recomendacion.replace("\n", "\n\n"))

st.divider()

col_grafico, col_tabla = st.columns([1, 1])

with col_grafico:
    st.subheader("Distribucion de clientes por riesgo")
    conteo_riesgo = datos["riesgo_actuarial"].value_counts()
    st.bar_chart(conteo_riesgo)

with col_tabla:
    st.subheader("Datos con clusters")
    st.caption(f"Accuracy de referencia del modelo: {accuracy:.2%}")
    st.dataframe(datos.head(20), use_container_width=True, hide_index=True)
