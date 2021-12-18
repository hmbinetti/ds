from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import streamlit as st
import spacy
from spacy.language import Language
from spacy import displacy
import pandas as pd
import pickle
import string
#Libreria para NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Analisis de sentimiento

from textblob import TextBlob
from textblob import Word


from nltk.sentiment.vader import SentimentIntensityAnalyzer # Analisis de sentimiento

from textblob import TextBlob
from textblob import Word

from util import load_model, process_text, get_svg, get_html, LOGO
from spacy.lang.en import English

# pip install spacy download en
# pip install spacy download en_core_web_sm

stop_words_eng = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation
spacy_model = 'en_core_web_sm'
nlp = load_model(spacy_model)

# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]

TOKEN_ATTRS = [ "idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit) - Trabajo Integrador Grupo 3 - DigitalHouse</span>"""

def prediccion(text):
    modelo_class = load_modelo()
    
    modelo_pred_state = st.info(f"Clasificando Texto '{modelo_class.best_estimator_[0]}'...")
    prediccion = modelo_class.predict([text])
    modelo_pred_state.empty()
    f"**El texto se clasifica como:** `{prediccion[0]}`"    
    st.success(f"**El texto se clasifica como         : `{(str.upper(prediccion[0]))}`**")
    return 
    
def load_modelo():
    pickle_file = 'Data/modelo_entrenado_app.pkl'
    with open(pickle_file, 'rb') as f:
        modelo_class = pickle.load(f)
    return modelo_class
        
def vader_texblob(text):   
        col1, col2 = st.columns(2)
        
        analizador = SentimentIntensityAnalyzer()
        polarity = analizador.polarity_scores(text)
        with col1:
            polarity_text = f"**VADER Polarity Score:** `{polarity['neg']}`"
            st.error(polarity_text)
            polarity_text = f"**VADER Polarity Score:** `{polarity['neu']}`"
            st.warning(polarity_text)
            polarity_text = f"**VADER Polarity Score:** `{polarity['pos']}`"
            st.success(polarity_text)
        with col2:
            st.image('Data/negativo.jpeg', width= 60)
            st.image('Data/neutro.jpeg', width= 60)
            st.image('Data/positivo.jpeg', width= 60)
        
        # Con TextBlob, buscar si hay algo mas para hacer 
        polarity =  TextBlob(text).sentiment.polarity 
        polarity_textblod = f"**TEXBLOD Polarity Score:** `{polarity}`"
        subjectivity = TextBlob(text).sentiment.subjectivity 
        subjectivity_text = f"**TEXBLOD Subjectivity Score:** `{subjectivity}`"
        
#         st.success(st.image('Data/positivo.jpeg', width= 100, caption={polarity_textblod}))
#         from PIL import Image
#         image = Image.open('sunrise.jpg')
        
        
        st.error(polarity_textblod)
        st.success(subjectivity_text)
        st.error(subjectivity_text)
        
        return

def visualize(
    models: Union[List[str], Dict[str, str]],
    default_text: str = "",
    default_model: Optional[str] = None,
    visualizers: List[str] = ["parser", "ner", "textcat", "similarity", "tokens"],
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    similarity_texts: Tuple[str, str] = ("apple", "orange"),
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_meta: bool = True,
    show_config: bool = True,
    show_visualizer_select: bool = False,
    show_pipeline_info: bool = True,
    sidebar_title: Optional[str] = "Analisis de Emociones y Semantica ",
    sidebar_description: Optional[str] = 'Trabajo Integrador Grupo 3',
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
    get_default_text: Callable[[Language], str] = None,
    spacy_model: str = 'en_core_web_md'
) -> None:
    """Embed the full visualizer with selected components."""

#     if st.config.get_option("theme.primaryColor") != color:
#         st.config.set_option("theme.primaryColor", color)

#         # Necessary to apply theming
#         st.experimental_rerun()

#     if show_logo:
#         st.sidebar.markdown(LOGO, unsafe_allow_html=True)
#     if sidebar_title:
#         st.sidebar.title(sidebar_title)
#     if sidebar_description:
#         st.sidebar.markdown(sidebar_description)

    # Allow both dict of model name / description as well as list of names
    model_names = models
    format_func = str
    if isinstance(models, dict):
        format_func = lambda name: models.get(name, name)
        model_names = list(models.keys())

    default_model_index = (
        model_names.index(default_model)
        if default_model is not None and default_model in model_names
        else 0
    )
    spacy_model = st.selectbox(
        "Modelo",
        model_names,
        index=default_model_index,
        key=f"{key}_visualize_models",
        format_func=format_func,
    )
    model_load_state = st.info(f"Cargando modelos '{spacy_model}'...")
    global nlp 
    nlp = load_model(spacy_model)
    model_load_state.empty()

    if show_pipeline_info:
        st.sidebar.subheader("Pipeline info")
        desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
        st.sidebar.markdown(desc, unsafe_allow_html=True)

    if show_visualizer_select:
        active_visualizers = st.sidebar.multiselect(
            "Visualizers",
            options=visualizers,
            default=list(visualizers),
            key=f"{key}_viz_select",
        )
    else:
        active_visualizers = visualizers

    default_text = (
        get_default_text(nlp) if get_default_text is not None else default_text
    )
    text = st.text_area("Ingrese Texto para analisis en idioma seleccionado", default_text, key=f"{key}_visualize_text")
    doc = process_text(spacy_model, text)
    
    if (spacy_model == 'en_core_web_sm') | (spacy_model == 'en_core_web_md') | (spacy_model == 'en_core_web_lg'):
       
        nlp = load_model(spacy_model)
        prediccion(text)
        vader_texblob(text)
    

    if "parser" in visualizers and "parser" in active_visualizers:
        visualize_parser(doc, key=key)
    if "ner" in visualizers and "ner" in active_visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, attrs=ner_attrs, key=key)
    if "textcat" in visualizers and "textcat" in active_visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers and "similarity" in active_visualizers:
        visualize_similarity(nlp, default_texts=similarity_texts, key=key)
    if "tokens" in visualizers and "tokens" in active_visualizers:
        visualize_tokens(doc, attrs=token_attrs, key=key)

    if show_json_doc or show_meta or show_config:
        st.header("Pipeline info")
        if show_json_doc:
            json_doc_exp = st.expander("JSON Doc")
            json_doc_exp.json(doc.to_json())

        if show_meta:
            meta_exp = st.expander("Pipeline meta.json")
            meta_exp.json(nlp.meta)

        if show_config:
            config_exp = st.expander("Pipeline config.cfg")
            config_exp.code(nlp.config.to_str())


def visualize_parser(
    doc: spacy.tokens.Doc,
    *,
    title: Optional[str] = "Análisis de Dependencia",
    key: Optional[str] = None,
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    cols = st.columns(4)
    split_sents = cols[0].checkbox(
        "Separa frases", value=True, key=f"{key}_parser_split_sents"
    )
    options = {
        "Collapse punct": cols[1].checkbox(
            "Contrae Puntuación", value=True, key=f"{key}_parser_collapse_punct"
        ),
        "collapse_phrases": cols[2].checkbox(
            "Contrae Frases", key=f"{key}_parser_collapse_phrases"
        ),
        "compact": cols[3].checkbox("Modo Compacto", key=f"{key}_parser_compact"),
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options, style="dep")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)


def visualize_ner(
    doc: Union[spacy.tokens.Doc, List[Dict[str, str]]],
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Entidades",
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
    manual: Optional[bool] = False,
) -> None:
    """Visualizer for named entities."""
    if title:
        st.header(title)

    if manual:
        if show_table:
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False."
            )
        if not isinstance(doc, list):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'list', not 'spacy.tokens.Doc'."
            )
    else:
        labels = labels or [ent.label_ for ent in doc.ents]

    if not labels:
        st.warning("The parameter 'labels' should not be empty or None.")
    else:
        exp = st.expander("Selecionar etiquetas entidades")
        label_select = exp.multiselect(
            "Etiquetas Entidades",
            options=labels,
            default=list(labels),
            key=f"{key}_ner_label_select",
        )
        html = displacy.render(
            doc,
            style="ent",
            options={"ents": label_select, "colors": colors},
            manual=manual,
        )
        style = "<style>mark.entity { display: inline-block }</style>"
        st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
        if show_table:
            data = [
                [str(getattr(ent, attr)) for attr in attrs]
                for ent in doc.ents
                if ent.label_ in label_select
            ]
            if data:
                df = pd.DataFrame(data, columns=attrs)
                st.dataframe(df)


def visualize_textcat(
    doc: spacy.tokens.Doc, *, title: Optional[str] = "Clasificación de Textos with spaCy"
) -> None:
    """Visualizer for text categories."""
    if title:
        st.header(title)
    st.markdown(f"> {doc.text}")
    
    st.markdown(doc.cats)
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


def visualize_similarity(
    nlp: spacy.language.Language,
    default_texts: Tuple[str, str] = ("apple", "Data"),
    *,
    threshold: float = 0.5,
    title: Optional[str] = "Vectoress & Similaridad",
    key: Optional[str] = None,
) -> None:
    """Visualizer for semantic similarity using word vectors."""
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if not meta.get("width", 0):
        st.warning("No existen vectores disponibles en el modelo.")
    else:
        cols = st.columns(2)
        text1 = cols[0].text_input(
            "Text or word 1", default_texts[0], key=f"{key}_similarity_text1"
        )
        text2 = cols[1].text_input(
            "Text or word 2", default_texts[1], key=f"{key}_similarity_text2"
        )
        doc1 = nlp.make_doc(text1)
        doc2 = nlp.make_doc(text2)
        similarity = doc1.similarity(doc2)
        similarity_text = f"**Score:** `{similarity}`"
        if similarity > threshold:
            st.success(similarity_text)
        else:
            st.error(similarity_text)

        exp = st.expander("Vector information")
        exp.code(meta)


def visualize_tokens(
    doc: spacy.tokens.Doc,
    *,
    attrs: List[str] = TOKEN_ATTRS,
    title: Optional[str] = "Token Atributos",
    key: Optional[str] = None,
) -> None:
    """Visualizer for token attributes."""
    if title:
        st.header(title)
    exp = st.expander("Seleccionar token atributos")
    selected = exp.multiselect(
        "Token Atributos",
        options=attrs,
        default=list(attrs),
        key=f"{key}_tokens_attr_select",
    )
    data = [[str(getattr(token, attr)) for attr in selected] for token in doc]
    df = pd.DataFrame(data, columns=selected)
    st.dataframe(df)
