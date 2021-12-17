# import spacy_streamlit - libreria para graficos en spaCy.
from visualizer import *

from class_def import *
import streamlit as st
import joblib  

global nlp, spacy_model, stop_words_eng

class cleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        # Limpiamos Texto
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Funcion para limpiar Texto - podriamos agregar mas cosas
def clean_text(text):
    
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()   

    # Poner el texto en minúsculas
    text = text.lower()
    
    # Remueve Acentos
    text= unidecode.unidecode(text)
        

    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    
    # Remueve los caracteres no ASCII
    text = text.replace(r'[^\x00-\x7F]+','')
    
    # Reemplaza espacios duplicados por un solo espacio
    text = text.replace(r'\s+',' ')
    
    # Reemplaza dos o mas puntos por uno
    text = text.replace(r'\.{2,}', ' ')
    
    # Reemplaza saltos de linea, etc por espacio
    text = text.replace('\n', "").replace('\r',"")
    
#     # Quitar las palabras que contengan números
#     text = [word for word in text if not any(c.isdigit() for c in word)]
    text = re.sub(r'[0-9]+', '', text)

    
#     # Quitar los tokens vacíos
#     text = [t for t in text if len(t) > 0]
    
#     # Quitar las palabras con sólo una letra
#     text = [t for t in text if len(t) > 1]

 
    #Remueve corchetes
    text = re.sub('\[[^]]*\]', '', text)

    text = re.sub(r'[^\W\S]','',text)
    
    return(text)

# Funcion para tokenizar utilizada en pipeline
def spacy_tokenizer(text):
    # Tokenizamos el texto en ingles
    
    mytokens = nlp(text)
#     # Tokenizamos el texto en Español
#     mytokens = nlp_esp(text)
#     st.error(s mytokens)

    # Lematizamos y convertimos en minusculas, aunque no es necesario porque lo podemos hacer en vectroizacion
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
#     print('esto largo lemma', mytokens)
    
    # Removemos Stop-Words
    mytokens = [ word for word in mytokens if word not in stop_words_eng and word not in punctuations ]
#     print('esto largo stop_word', mytokens)
    
    return mytokens

def visualize_sidebar(
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
    sidebar_title: Optional[str] = '',
    sidebar_description: Optional[str] = '',
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
    get_default_text: Callable[[Language], str] = None,
) -> None:
    """Embed the full visualizer with selected components."""
#     global nlp, spacy_model
    
    if st.config.get_option("theme.primaryColor") != color:
        st.config.set_option("theme.primaryColor", color)

        # Necessary to apply theming
        st.experimental_rerun()

    if show_logo:
        st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        st.sidebar.title(sidebar_title)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

#     # Allow both dict of model name / description as well as list of names
#     model_names = models
#     format_func = str
#     if isinstance(models, dict):
#         format_func = lambda name: models.get(name, name)
#         model_names = list(models.keys())

#     default_model_index = (
#         model_names.index(default_model)
#         if default_model is not None and default_model in model_names
#         else 0
#     )
#     spacy_model = st.sidebar.selectbox(
#         "Modelo",
#         model_names,
#         index=default_model_index,
#         key=f"{key}_visualize_models",
#         format_func=format_func,
#     )
#     model_load_state = st.info(f"Cargando modelos '{spacy_model}'...")
#     nlp = load_model(spacy_model)
#     model_load_state.empty()

#     if show_pipeline_info:
#         st.sidebar.subheader("Pipeline info")
#         desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
#         st.sidebar.markdown(desc, unsafe_allow_html=True)

    st.sidebar.markdown(
        FOOTER,
        unsafe_allow_html=True,
    )
    
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
    

def scraper(scraper_btton, scraper_btton_options, hash_select ):
    if scraper_btton == scraper_btton_options['opcion2']:
        df_scraper = API_Twitter(hash_select)
    return df_scraper
    
def API_Twitter(hash_list):
    
    twitter_file_loc = 'Data\datos_twitter.json'

    CONSUMER_TOKEN = "RlqYWgIGqEIcM0VDr1KipSxgS"
    CONSUMER_SECRET = "lfR9byA6OVr0ldECQPYuBBYfWcWgSfI7UVkqJZHDPhRQPdGC7K"
    ACCESS_TOKEN = "19623284-SHznE8xBiPWeJ7UA5UP30KeXqRdth5WC4jcyQX2fO"
    ACCESS_TOKEN_SECRET = "PXq4RG7AEE6yQcLzxSX7JWYcvjFUv2vv7wR025rGdxSXk"
    
    auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    
    try:
        api.verify_credentials()
        st.success("Autenticación OK")
    except:
        st.error("Error en la Autenticación")
    
    my_bar = st.progress(0)
    percent_complete = 0
    with open(twitter_file_loc, 'w', encoding='utf-8') as twitter_file:
        lista_json = []
        for hash in hash_list:
            lista_tweets = api.search_tweets(q=hash + '-filter:retweets', lang="en", count=100)
            for tweet in lista_tweets:
                    lista_json.append(tweet._json)
            percent_complete += int(100/len(hash_list))
            my_bar.progress(percent_complete)
        #lista_json[0]
        json.dump(lista_json, twitter_file, ensure_ascii=True, indent=4)
        my_bar.progress(100)
        df_tweet = pd.DataFrame(lista_json)
            
#     st.dataframe(df_tweet['text'])
    return df_tweet

def do_wordcloud(df):
    
    df=df.progress_apply(lambda x: clean_text(x))
    
    corpus = " ".join([text for text in df])
    stopwords_cloud = set(STOPWORDS)
    wordcloud = WordCloud(background_color = 'cyan', stopwords = stopwords_cloud, width = 1200, height = 800).generate(corpus)
    st.subheader('WordCloud')
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (15, 15)
#     plt.title('Word Cloud Reviews', fontsize = 30)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()
    st.pyplot(fig)
    return

# Cuerpo de la APP - Streamlit


def main():
   
    # Parametros de la aplicacion
    app_mode_options = { 'opcion1': 'Text Scraping', 'opcion2': 'Analisis x Archivo', 'opcion3': 'Analisis de Emociones', 'opcion4' : 'Equipo'}
    scraper_btton_options = { 'opcion1': 'Instagram', 'opcion2': 'Twitter', 'opcion3': 'Facebook', 'opcion4' : 'Reddit'}
    hashtag = ['emotion', 'sentiment', 'feel', 'regret', 'sorry', 'sad']
    app_mode_list = ['Text Scraping', 'Analisis x Archivo', 'Analisis de Emociones', 'Equipo']
    modelos = ["en_core_web_sm", "en_core_web_lg", "en_core_web_md", "es_core_news_lg", 'it_core_news_md', "fr_core_news_md", "zh_core_web_md"]
    file_type = ['csv', 'json']

    # Seteamos Titulos y header
    st.title('Analisis de Emociones y Sentimientos')
    st.header('Sintaxis y Semantica')

    # Seteamos el modo de operacion
    app_mode = st.sidebar.selectbox(
            "Modo",
            app_mode_list,
            )
    #         key=f"{key}_visualize_models",
    #         format_func=format_func,

#     model_load_state = st.info(f"Cargando modelos '{spacy_model}'...")
#     nlp = load_model(spacy_model)
#     model_load_state.empty()

    visualize_sidebar(modelos)
    
    if app_mode == app_mode_options['opcion4']:
        st.markdown('Valentina Scarano')
        st.markdown('Juan Manuel Gonzalez Prates')
        st.markdown('Facundo Guinazu')
        st.markdown('Matias Taurizano')
        st.markdown('Horacio Binetti')
    elif app_mode == app_mode_options['opcion2']:
        uploaded_file = st.file_uploader("Elija un archivo .json", type = file_type, help='Archivo de Text Wrapping de Twitter para analizar emociones')
        btton = st.button('Inicia')
        if btton == True:
            model_load_state = st.info(f"Prediciendo '{scraper_btton_options}'...")
            df_scraper = pd.read_json(uploaded_file)
            modelo_class = load_modelo()
            prediccion = modelo_class.predict(df_scraper['text'])
            df_scraper_pred = pd.concat([df_scraper, pd.DataFrame(prediccion, columns=["predict"])], axis=1)
            st.dataframe(df_scraper_pred[['text', 'predict']])
            model_load_state.empty()
            do_wordcloud(df_scraper['text'])
        
    elif app_mode == app_mode_options['opcion1']:
        exp = st.expander("Tema para Sraping")
        hash_select = exp.multiselect(
            "Temas",
            options=hashtag,
            default=list(hashtag),
        
        )
        
        scraping = st.radio(
         "que red desea scrapear",
         (scraper_btton_options['opcion1'], scraper_btton_options['opcion2'], scraper_btton_options['opcion3'], scraper_btton_options['opcion4']))
        btton = st.button('Inicia')
        if btton == True:
            df_scraper = scraper(scraping, scraper_btton_options, hash_select)
            model_load_state = st.info(f"Prediciendo '{scraper_btton_options}'...")
   
    
            modelo_class = load_modelo()
            prediccion = modelo_class.predict(df_scraper['text'])
#             df_scraper['predict'] = df_scraper['text'].apply(lambda x: modelo_class.predict(x))
            df_scraper_pred = pd.concat([df_scraper, pd.DataFrame(prediccion, columns=["predict"])], axis=1)
            st.dataframe(df_scraper_pred[['text', 'predict']])
            do_wordcloud(df_scraper['text'])
            model_load_state.empty()
    else:

        default_text = "Data science is a multidisciplinary approach to extracting actionable insights from the large and ever-increasing volumes of data collected and created by today’s organizations. Data science encompasses preparing data for analysis and processing, performing advanced data analysis, and presenting the results to reveal patterns and enable stakeholders to draw informed conclusions."

        visualize(modelos, default_text, spacy_model='en_core_web_md' )

    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    # sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelBinarizer


    from sklearn.svm import SVC
    from sklearn import metrics
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression,SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.decomposition import TruncatedSVD
    from xgboost.sklearn import XGBClassifier
    from catboost import CatBoostClassifier
    import lightgbm as lgb

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
    
    from wordcloud import WordCloud,STOPWORDS
    
    # Libreria para barras de progreso
    from tqdm.notebook import tqdm_notebook
    tqdm_notebook.pandas()
    from tqdm import tqdm, tqdm_notebook

    from bs4 import BeautifulSoup
    
    import tweepy
    import json

#     global nlp
    
    stop_words_eng = spacy.lang.en.stop_words.STOP_WORDS
    punctuations = string.punctuation
    spacy_model = 'en_core_web_sm'
   
    st.set_page_config(
    page_title="Analisis de Sentimiento y Emociones",
    page_icon="🙂",
    layout="centered",
    initial_sidebar_state="expanded",)
    main()