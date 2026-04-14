import streamlit as st
import re
import spacy
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
from itertools import combinations

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Análise Completa de Sentimentos", layout="wide")

PRIMARY = "#8B0000"
SECONDARY = "#5A0000"
BG = "#0E0E0E"

# =========================================================
# CSS
# =========================================================
st.markdown(f"""
<style>
.stApp {{background-color:{BG};}}
h1,h2,h3,h4 {{color:{PRIMARY};}}
p,label,div {{color:white;}}
textarea {{
    background-color:#1A1A1A !important;
    color:white !important;
}}
button {{
    background-color:{PRIMARY} !important;
    color:white !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODELOS
# =========================================================
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_models():
    bert = pipeline("sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment")

    roberta = pipeline("sentiment-analysis",
                       model="cardiffnlp/twitter-roberta-base-sentiment")
    return bert, roberta

nlp = load_spacy()
bert_model, roberta_model = load_models()

# =========================================================
# STOPWORDS WORDCLOUD (corrigido)
# =========================================================
CUSTOM_STOP = set(STOPWORDS)
CUSTOM_STOP.update(list(nlp.Defaults.stop_words))
CUSTOM_STOP.update(["e","a","o","de","da","do"])

# =========================================================
# LIMPEZA
# =========================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# =========================================================
# PREPROCESSAMENTO COMPLETO
# =========================================================
def preprocess(text):
    text = clean_text(text)
    doc = nlp(text)

    tokens = [t.text for t in doc if t.is_alpha]
    tokens_no_stop = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t)>2]
    stop_removed = [t.text for t in doc if t.is_stop]

    return tokens, tokens_no_stop, stop_removed, doc

# =========================================================
# REDE COOCORRENCIA
# =========================================================
def cooccurrence(tokens, window=4, min_freq=2):
    pairs = Counter()
    for i in range(len(tokens)-window):
        window_tokens = tokens[i:i+window]
        for w1, w2 in combinations(set(window_tokens), 2):
            pairs[(w1,w2)] += 1
    return {k:v for k,v in pairs.items() if v >= min_freq}

# =========================================================
# MAP SENTIMENTOS
# =========================================================
def map_bert(label):
    stars = int(label.split()[0])
    if stars <= 2:
        return "Negativo"
    elif stars == 3:
        return "Neutro"
    else:
        return "Positivo"

def map_roberta(label):
    if label == "LABEL_0":
        return "Negativo"
    elif label == "LABEL_1":
        return "Neutro"
    else:
        return "Positivo"

# =========================================================
# SENTENÇAS
# =========================================================
def sentence_sentiments(doc):
    bert_labels=[]
    roberta_labels=[]
    groups={"Positivo":[],"Neutro":[],"Negativo":[]}

    for sent in doc.sents:
        s=sent.text.strip()
        if len(s)<3:
            continue

        b=map_bert(bert_model(s[:512])[0]["label"])
        r=map_roberta(roberta_model(s[:512])[0]["label"])

        bert_labels.append(b)
        roberta_labels.append(r)
        groups[b].append(s)

    return bert_labels, roberta_labels, groups

# =========================================================
# WORDCLOUD CORRIGIDO
# =========================================================
def plot_wordcloud(words):
    if not words:
        st.warning("Sem palavras suficientes")
        return

    text=" ".join(words)
    wc=WordCloud(
        width=900,
        height=400,
        background_color=BG,
        stopwords=CUSTOM_STOP,
        colormap="Reds"
    ).generate(text)

    st.image(wc.to_array())

# =========================================================
# TITULO
# =========================================================
st.title("🧠 Plataforma Científica de Análise de Sentimentos")

text_input=st.text_area("Cole o texto",height=220)

# =========================================================
# EXECUÇÃO
# =========================================================
if text_input:

    tokens, tokens_no_stop, stop_removed, doc=preprocess(text_input)
    bert_sent, rob_sent, groups=sentence_sentiments(doc)

    tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
        "📊 Visão Geral",
        "🧹 Pré-processamento",
        "📈 Estatística",
        "🌐 Rede Semântica",
        "☁️ Nuvens",
        "🤖 Modelos",
        "📚 Documentação & Referências"
    ])

    # =====================================================
    # VISÃO GERAL
    # =====================================================
    with tab1:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total palavras",len(tokens))
        c2.metric("Sem stopwords",len(tokens_no_stop))
        c3.metric("Stopwords removidas",len(stop_removed))
        c4.metric("Sentenças",len(list(doc.sents)))

    # =====================================================
    # PRE PROCESSAMENTO (agora explicativo)
    # =====================================================
    with tab2:

        st.subheader("O que é o pré-processamento textual?")
        st.write("""
O pré-processamento transforma texto bruto em dados analisáveis.
Nesta etapa ocorrem:

Tokenização: separação do texto em unidades linguísticas  
Remoção de stopwords: retirada de palavras sem valor semântico  
Lematização: redução das palavras à sua forma base  
Limpeza textual: remoção de ruído, pontuação e URLs  
Filtragem de tamanho mínimo de palavras  
""")

        st.subheader("Tokens com stopwords")
        st.write(tokens[:200])

        st.subheader("Tokens lematizados sem stopwords")
        st.write(tokens_no_stop[:200])

    # =====================================================
    # ESTATISTICA
    # =====================================================
    with tab3:

        lengths=[len(t) for t in tokens_no_stop]
        df=pd.DataFrame({"Comprimento":lengths})

        st.subheader("Distribuição do tamanho das palavras")
        st.plotly_chart(px.histogram(df,x="Comprimento",
                                     color_discrete_sequence=[PRIMARY]),
                        use_container_width=True)

        st.subheader("Boxplot")
        st.plotly_chart(px.box(df,y="Comprimento",
                               color_discrete_sequence=[PRIMARY]),
                        use_container_width=True)

        st.subheader("Palavras mais frequentes")
        freq=Counter(tokens_no_stop).most_common(20)
        df_freq=pd.DataFrame(freq,columns=["Palavra","Freq"])
        st.plotly_chart(px.bar(df_freq,x="Palavra",y="Freq",
                               color_discrete_sequence=[PRIMARY]),
                        use_container_width=True)

    # =====================================================
    # REDE VISUAL MELHORADA
    # =====================================================
    with tab4:

        cooc=cooccurrence(tokens_no_stop)

        if cooc:
            G=nx.Graph()
            for (w1,w2),v in cooc.items():
                G.add_edge(w1,w2,weight=v)

            fig,ax=plt.subplots(figsize=(9,7))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)

            pos=nx.spring_layout(G,k=0.6,seed=42)

            nx.draw_networkx_nodes(G,pos,node_color=PRIMARY,node_size=700)
            nx.draw_networkx_edges(G,pos,edge_color="black",width=1.5)
            nx.draw_networkx_labels(G,pos,font_color="white")

            ax.set_title("Rede de Coocorrência",color="white")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Rede insuficiente")

    # =====================================================
    # NUVENS
    # =====================================================
    with tab5:
        st.subheader("Nuvem geral")
        plot_wordcloud(tokens_no_stop)

        c1,c2,c3=st.columns(3)
        if c1.button("Positivo"):
            plot_wordcloud(groups["Positivo"])
        if c2.button("Neutro"):
            plot_wordcloud(groups["Neutro"])
        if c3.button("Negativo"):
            plot_wordcloud(groups["Negativo"])

    # =====================================================
    # MODELOS
    # =====================================================
    with tab6:

        df_model=pd.DataFrame({
            "BERT":Counter(bert_sent),
            "RoBERTa":Counter(rob_sent)
        }).fillna(0)

        df_model=df_model.reset_index().rename(columns={"index":"Sentimento"})

        st.plotly_chart(px.bar(df_model,x="Sentimento",
                               y=["BERT","RoBERTa"],
                               barmode="group",
                               color_discrete_sequence=[PRIMARY,SECONDARY]),
                        use_container_width=True)

    # =====================================================
    # DOCUMENTAÇÃO + REFERENCIAS
    # =====================================================
    with tab7:

        st.header("Documentação Metodológica")

        st.write("""
Este sistema foi desenvolvido em Python utilizando Streamlit para construção da interface interativa.
O ambiente de desenvolvimento utilizado é o Visual Studio Code.

O processamento linguístico é realizado com spaCy.
A análise de sentimentos utiliza modelos Transformer pré-treinados:
BERT multilíngue e RoBERTa.

Etapas da análise:
1 Limpeza textual
2 Tokenização
3 Remoção de stopwords
4 Lematização
5 Estatística descritiva
6 Modelagem de sentimentos
7 Visualização semântica em rede
""")

        st.header("Referências")

        st.write("""
Bird S., Klein E., Loper E. Natural Language Processing with Python. O'Reilly Media.

Jurafsky D., Martin J. Speech and Language Processing. Pearson.

Devlin J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2019.

Liu Y. et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. 2019.

McKinney W. Data Structures for Statistical Computing in Python. 2010.

Pedregosa F. Scikit-learn Machine Learning in Python. 2011.
""")

