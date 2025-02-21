import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configuração da página
st.set_page_config(layout="wide", page_title="Dashboard de Análise de Crédito")

# Estilo Global
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
        }
        .stSelectbox, .stNumberInput {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# 🔹 Carregar modelo e scaler
modelo = joblib.load("modelo_credit_approval.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 Carregar e preparar os dados
df = pd.read_csv("crx.data", header=None)
column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Class']
df.columns = column_names

# Tratamento dos dados
df.replace("?", np.nan, inplace=True)
for col in df:
    if df[col].dtypes == 'object':
        df[col] = df[col].fillna(df[col].value_counts().index[0])
        df[col] = pd.factorize(df[col])[0]
    else:
        df[col] = df[col].astype(float).fillna(df[col].mean())

# 🎯 **Dashboard Principal**
st.title("📊 Dashboard Interativo de Análise de Crédito")

# 📌 **Sidebar para Entrada de Dados**
st.sidebar.header("🔍 Insira os dados do Cliente:")
inputs = []
for col in df.columns[:-1]:  # Ignora a coluna 'Class'
    valor = st.sidebar.number_input(f"{col}", value=float(df[col].mean()))
    inputs.append(valor)

# Transformação e Previsão
inputs = np.array(inputs).reshape(1, -1)
inputs_scaled = scaler.transform(inputs)
previsao = modelo.predict(inputs_scaled)
resultado = "🟢 Aprovado" if previsao[0] == 1 else "🔴 Negado"

# Exibir resultado
st.markdown(f"## 🏦 **Resultado da Análise:** {resultado}")

# 🚀 **Botão para reexecutar previsão**
if st.button("🔄 Recalcular Previsão"):
    st.rerun()

# Criar Tabs para melhor navegação dos gráficos
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribuição das Variáveis", "📈 Matriz de Confusão", "📉 Análise de Correlação", "📋 Estatísticas Gerais"])

# 📊 **1. Distribuição das Variáveis**
with tab1:
    st.subheader("📊 Distribuição das Variáveis")
    selected_feature = st.selectbox("Escolha uma variável:", df.columns[:-1])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df, x=selected_feature, hue="Class", bins=30, kde=True, alpha=0.6, palette="coolwarm", ax=ax)
    st.pyplot(fig)

# 📈 **2. Matriz de Confusão**
with tab2:
    st.subheader("📈 Matriz de Confusão")
    y_real = df["Class"]
    y_pred = modelo.predict(scaler.transform(df.drop("Class", axis=1)))
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)

# 📉 **3. Análise de Correlação**
with tab3:
    st.subheader("📉 Mapa de Correlação entre Variáveis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# 📋 **4. Estatísticas Gerais**
with tab4:
    st.subheader("📋 Estatísticas Gerais do Dataset")
    st.write(df.describe())

    # Relatório de Classificação
    st.subheader("📑 Relatório de Performance do Modelo")
    report = classification_report(y_real, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, subset=["precision", "recall", "f1-score"], color="lightblue"))

# 🚀 Finalização
st.markdown("🎯 **Dashboard interativo para análise de crédito com Machine Learning!** 🚀")
