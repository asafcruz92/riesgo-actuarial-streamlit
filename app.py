import pandas as pd
import streamlit as st
from openai import OpenAI
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
                    solver="liblinear",
                ),
            ),
        ]
    )

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)

    modelo.fit(X, y)

    return modelo, accuracy


def obtener_api_key():
    api_key_secrets = st.secrets.get("OPENAI_API_KEY", "")

    if api_key_secrets:
        return api_key_secrets

    return st.sidebar.text_input(
        "API key para recomendaciones",
        type="password",
        help="Si la app esta publicada, lo ideal es guardar esta clave en los secrets de Streamlit.",
    )


def generar_recomendacion(api_key, riesgo, cliente):
    if not api_key:
        return (
            "Para generar recomendaciones con API, se debe configurar una API key. "
            "Mientras tanto, el modelo ya puede calcular el nivel de riesgo actuarial."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
    Eres un asistente actuarial academico. Redacta recomendaciones breves, sencillas y prudentes
    para un cliente de seguro medico con este perfil:

    Riesgo actuarial predicho: {riesgo}
    Edad: {cliente["age"]}
    Sexo: {cliente["sex"]}
    BMI: {cliente["bmi"]}
    Hijos: {cliente["children"]}
    Fumador: {cliente["smoker"]}
    Region: {cliente["region"]}
    Cargos medicos estimados: {cliente["charges"]}

    Escribe 3 recomendaciones en espanol claro. No indiques que se debe aprobar o rechazar una poliza.
    """

    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Responde de forma breve, clara y responsable."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return respuesta.choices[0].message.content


datos = cargar_datos()
modelo, accuracy = entrenar_modelo(datos)
api_key = obtener_api_key()

st.title("Prediccion de riesgo actuarial - Su nombre - PCAF-03")
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

    st.subheader("Recomendaciones generadas con API")
    try:
        recomendacion = generar_recomendacion(api_key, riesgo, cliente.iloc[0].to_dict())
        st.write(recomendacion)
    except Exception as error:
        st.warning("No se pudo generar la recomendacion con API.")
        st.caption(f"Detalle tecnico: {error}")

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
