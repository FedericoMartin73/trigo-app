"""App del TP: producción de trigo en Argentina (1927-2025).

Reutiliza la limpieza (Fase 3) y el modelo (Fases 4 y 5) del notebook.
Para correrla localmente:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Trigo en Argentina", page_icon="🌾", layout="wide")

PREDICTORAS = ["anio", "superficie_sembrada_ha", "produccion_anterior", "rendimiento_anterior", "provincia_cod"]
OBJETIVO = "produccion_tm"
ANIO_CORTE = 2014  # entrenamiento hasta 2014, prueba desde 2015


# -----------------------------------------------------------------------------
# Datos y modelo (se calculan una sola vez y quedan en caché)
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv("data/trigo-serie-1927-2025.csv")

    # Limpieza de la Fase 3: sin nulos ni ceros en las variables de cosecha
    cols = ["superficie_cosechada_ha", "produccion_tm", "rendimiento_kgxha"]
    df = df.dropna(subset=cols + ["departamento", "departamento_id"])
    df = df[(df[cols] != 0).all(axis=1)]

    # Fase 4: un registro por departamento y año
    datos = (
        df.groupby(["provincia", "departamento", "anio"], as_index=False)
        [["superficie_sembrada_ha", "superficie_cosechada_ha", "produccion_tm"]]
        .sum()
    )
    datos["rendimiento_kgxha"] = datos["produccion_tm"] / datos["superficie_cosechada_ha"] * 1000

    # Datos de la campaña anterior de cada departamento
    por_depto = datos.groupby(["provincia", "departamento"])
    datos["anio_anterior"] = por_depto["anio"].shift(1)
    datos["produccion_anterior"] = por_depto["produccion_tm"].shift(1)
    datos["rendimiento_anterior"] = por_depto["rendimiento_kgxha"].shift(1)
    datos["provincia_cod"] = datos["provincia"].astype("category").cat.codes
    return datos


@st.cache_resource
def entrenar():
    datos = cargar_datos()
    modelo_df = datos[datos["anio"] - datos["anio_anterior"] == 1]
    train = modelo_df[modelo_df["anio"] <= ANIO_CORTE]
    test = modelo_df[modelo_df["anio"] > ANIO_CORTE]

    lineal = LinearRegression().fit(train[PREDICTORAS], train[OBJETIVO])
    bosque = RandomForestRegressor(random_state=42, n_jobs=-1).fit(train[PREDICTORAS], train[OBJETIVO])

    predicciones = {
        "Regla del año anterior": test["produccion_anterior"],
        "Regresión lineal": lineal.predict(test[PREDICTORAS]),
        "Random Forest": bosque.predict(test[PREDICTORAS]),
    }

    # Modelo final: Random Forest reentrenado con todos los años
    final = RandomForestRegressor(random_state=42, n_jobs=-1).fit(modelo_df[PREDICTORAS], modelo_df[OBJETIVO])
    return test, predicciones, bosque, final


def error_pct(real, pred) -> float:
    return abs(real - pred).sum() / real.sum() * 100


datos = cargar_datos()
test, predicciones, bosque, modelo_final = entrenar()
ultimo_anio = int(datos["anio"].max())

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.title("🌾 Producción de trigo en Argentina")
st.caption(
    "Trabajo Práctico · INGENIERÍA DEL SOFTWARE - Ciencia de Datos · UGR. "
    "Datos: Secretaría de Agricultura, Ganadería y Pesca (datos.gob.ar), campañas 1927–2025."
)

tab_hist, tab_zonas, tab_pron, tab_modelo = st.tabs(
    ["📈 Evolución histórica", "📍 Zonas clave", "🔮 Pronóstico", "🧪 Cómo funciona el modelo"]
)

# -----------------------------------------------------------------------------
# 1. Evolución histórica
# -----------------------------------------------------------------------------
with tab_hist:
    st.subheader("¿Cómo cambió el cultivo en 100 años?")
    c1, c2 = st.columns(2)
    provincias = sorted(datos["provincia"].unique())
    zona = c1.selectbox("Zona", ["Todo el país"] + provincias)
    variable = c2.radio(
        "Variable",
        ["Producción", "Superficie sembrada", "Rendimiento"],
        horizontal=True,
    )

    sub = datos if zona == "Todo el país" else datos[datos["provincia"] == zona]
    anual = sub.groupby("anio")[["produccion_tm", "superficie_sembrada_ha", "superficie_cosechada_ha"]].sum()
    serie = {
        "Producción": ("Millones de toneladas", anual["produccion_tm"] / 1e6),
        "Superficie sembrada": ("Millones de hectáreas", anual["superficie_sembrada_ha"] / 1e6),
        "Rendimiento": ("kg por hectárea", anual["produccion_tm"] / anual["superficie_cosechada_ha"] * 1000),
    }
    eje, valores = serie[variable]
    graf = valores.rename(eje).reset_index()

    st.altair_chart(
        alt.Chart(graf)
        .mark_line(point=alt.OverlayMarkDef(size=20))
        .encode(
            x=alt.X("anio:Q", title="Campaña", axis=alt.Axis(format="d"), scale=alt.Scale(nice=False)),
            y=alt.Y(f"{eje}:Q", title=eje),
            tooltip=[alt.Tooltip("anio:Q", title="Campaña", format="d"), alt.Tooltip(f"{eje}:Q", format=",.2f")],
        )
        .properties(height=380),
    )
    st.info(
        "Hoy se siembra **menos** que en los años 20 (entre 5 y 6,5 millones de hectáreas contra 8,6), "
        "pero el **rendimiento** pasó de unos 900 a más de 3.000 kg/ha, con el mayor salto desde los años 80 "
        "(nuevas semillas, fertilización, siembra directa). Por eso la producción casi se triplicó. "
        "Las caídas bruscas, como la de 2022, corresponden a sequías."
    )

# -----------------------------------------------------------------------------
# 2. Zonas clave
# -----------------------------------------------------------------------------
with tab_zonas:
    st.subheader("¿Dónde se produce el trigo?")
    desde, hasta = st.slider(
        "Período", int(datos["anio"].min()), ultimo_anio, (ultimo_anio - 9, ultimo_anio)
    )
    periodo = datos[datos["anio"].between(desde, hasta)]
    total = periodo["produccion_tm"].sum()

    prov = (
        periodo.groupby("provincia")["produccion_tm"].sum().sort_values(ascending=False).reset_index()
    )
    prov["Participación (%)"] = prov["produccion_tm"] / total * 100

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Provincias: participación en la producción total**")
        st.altair_chart(
            alt.Chart(prov.head(10))
            .mark_bar()
            .encode(
                x=alt.X("Participación (%):Q"),
                y=alt.Y("provincia:N", sort="-x", title=None),
                tooltip=["provincia", alt.Tooltip("Participación (%):Q", format=".1f")],
            )
            .properties(height=360),
        )
    with c2:
        st.markdown("**Top 10 departamentos**")
        deptos = (
            periodo.groupby(["provincia", "departamento"])["produccion_tm"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        deptos["Participación (%)"] = (deptos["produccion_tm"] / total * 100).round(1)
        deptos = deptos.rename(columns={"provincia": "Provincia", "departamento": "Departamento"})
        st.dataframe(
            deptos[["Departamento", "Provincia", "Participación (%)"]],
            hide_index=True,
            width="stretch",
        )

    top3 = prov.head(3)
    st.info(
        f"Entre {desde} y {hasta}, **{', '.join(top3['provincia'])}** concentraron el "
        f"**{top3['Participación (%)'].sum():.0f} %** de la producción del país."
    )

# -----------------------------------------------------------------------------
# 3. Pronóstico
# -----------------------------------------------------------------------------
with tab_pron:
    st.subheader(f"Pronóstico de la campaña {ultimo_anio + 1}/{ultimo_anio + 2}")
    st.write(
        "Todavía no se sabe cuánto se va a sembrar, así que podés probar escenarios: "
        f"mové el control para indicar cuánto cambia la superficie sembrada respecto a {ultimo_anio}."
    )
    cambio = st.slider("Cambio en la superficie sembrada (%)", -30, 30, 0, step=5)

    ultimo = datos[datos["anio"] == ultimo_anio]
    proximo = pd.DataFrame({
        "anio": ultimo_anio + 1,
        "superficie_sembrada_ha": ultimo["superficie_sembrada_ha"] * (1 + cambio / 100),
        "produccion_anterior": ultimo["produccion_tm"],
        "rendimiento_anterior": ultimo["rendimiento_kgxha"],
        "provincia_cod": ultimo["provincia_cod"],
        "provincia": ultimo["provincia"],
    })
    proximo["prediccion"] = modelo_final.predict(proximo[PREDICTORAS])

    pais_test = pd.DataFrame({"anio": test["anio"], "real": test[OBJETIVO], "pred": predicciones["Random Forest"]})
    pais_test = pais_test.groupby("anio").sum()
    error_nac = error_pct(pais_test["real"], pais_test["pred"])

    pronostico = proximo["prediccion"].sum() / 1e6
    real_ultimo = ultimo["produccion_tm"].sum() / 1e6
    c1, c2, c3 = st.columns(3)
    c1.metric("Producción estimada", f"{pronostico:.1f} Mt", f"{pronostico - real_ultimo:+.1f} Mt vs. {ultimo_anio}")
    c2.metric("Rango probable", f"{pronostico * (1 - error_nac / 100):.1f} – {pronostico * (1 + error_nac / 100):.1f} Mt")
    c3.metric("Superficie sembrada supuesta", f"{proximo['superficie_sembrada_ha'].sum() / 1e6:.1f} M ha")

    st.markdown("**Producción estimada por provincia**")
    por_prov = (proximo.groupby("provincia")["prediccion"].sum() / 1e6).sort_values(ascending=False).head(8)
    st.altair_chart(
        alt.Chart(por_prov.rename("Millones de toneladas").reset_index())
        .mark_bar()
        .encode(
            x=alt.X("Millones de toneladas:Q"),
            y=alt.Y("provincia:N", sort="-x", title=None),
            tooltip=["provincia", alt.Tooltip("Millones de toneladas:Q", format=".2f")],
        )
        .properties(height=300),
    )
    st.warning(
        f"El rango usa el error típico del modelo a nivel nacional (±{error_nac:.0f} %). "
        "El modelo **no tiene datos de clima**: una sequía o una campaña excepcional pueden quedar fuera de ese rango."
    )

# -----------------------------------------------------------------------------
# 4. Cómo funciona el modelo
# -----------------------------------------------------------------------------
with tab_modelo:
    st.subheader("¿Qué tan bien predice?")
    st.write(
        "Entrenamos con las campañas hasta 2014 y probamos con 2015–2025, años que el modelo no vio. "
        "Comparamos el Random Forest con una regresión lineal y con una regla simple: "
        "*\"este año se produce lo mismo que el año pasado\"*."
    )

    filas = []
    for nombre, pred in predicciones.items():
        pais = pd.DataFrame({"anio": test["anio"], "real": test[OBJETIVO], "pred": pred}).groupby("anio").sum()
        filas.append({
            "Modelo": nombre,
            "Error promedio por depto (t)": round(mean_absolute_error(test[OBJETIVO], pred)),
            "R²": round(r2_score(test[OBJETIVO], pred), 2),
            "Error % nacional": round(error_pct(pais["real"], pais["pred"]), 1),
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.caption(
        "**R²**: qué parte de la variación explica el modelo (más cerca de 1, mejor). "
        "**Error % nacional**: cuánto le erra al total del país, sumando todos los departamentos."
    )

    st.markdown("**Producción nacional: real vs. predicha por el Random Forest**")
    comp = (pais_test / 1e6).rename(columns={"real": "Real", "pred": "Predicción"}).reset_index()
    comp = comp.melt("anio", var_name="Serie", value_name="Millones de toneladas")
    st.altair_chart(
        alt.Chart(comp)
        .mark_line(point=True)
        .encode(
            x=alt.X("anio:Q", title="Campaña", axis=alt.Axis(format="d"), scale=alt.Scale(nice=False)),
            y=alt.Y("Millones de toneladas:Q"),
            color=alt.Color("Serie:N", title=None, scale=alt.Scale(domain=["Real", "Predicción"], range=["#1f77b4", "#ff7f0e"])),
            tooltip=[alt.Tooltip("anio:Q", title="Campaña", format="d"), "Serie", alt.Tooltip("Millones de toneladas:Q", format=".1f")],
        )
        .properties(height=340),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**¿Qué variables pesan más?**")
        nombres = {
            "anio": "Año",
            "superficie_sembrada_ha": "Superficie sembrada",
            "produccion_anterior": "Producción año anterior",
            "rendimiento_anterior": "Rendimiento año anterior",
            "provincia_cod": "Provincia",
        }
        imp = pd.DataFrame({
            "Variable": [nombres[v] for v in PREDICTORAS],
            "Importancia (%)": bosque.feature_importances_ * 100,
        })
        st.altair_chart(
            alt.Chart(imp)
            .mark_bar()
            .encode(
                x=alt.X("Importancia (%):Q"),
                y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
                tooltip=["Variable", alt.Tooltip("Importancia (%):Q", format=".0f")],
            )
            .properties(height=220),
        )
    with c2:
        st.markdown("**Limitaciones**")
        st.markdown(
            "- No usa datos de clima, así que no anticipa sequías ni campañas récord "
            "(ver 2022 y 2025 en el gráfico).\n"
            "- Se apoya mucho en la campaña anterior.\n"
            "- Hay que reentrenarlo cada campaña con los datos nuevos."
        )
