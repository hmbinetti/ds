# Fit y Transform usuando spaCy 
# Creamos la clase cleaner para preprosemaniento de texto a utilizar en pipeline

from sklearn.base import TransformerMixin
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re,string,unicodedata
import unicodedata
import unidecode
    
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
    import spacy
    stop_words_eng = spacy.lang.en.stop_words.STOP_WORDS
    mytokens = nlp(text)
#     # Tokenizamos el texto en Español
#     mytokens = nlp_esp(text)
#     print('esto largo el parser', mytokens)

    # Lematizamos y convertimos en minusculas, aunque no es necesario porque lo podemos hacer en vectroizacion
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
#     print('esto largo lemma', mytokens)
    
    # Removemos Stop-Words
    
    mytokens = [ word for word in mytokens if word not in stop_words_eng and word not in punctuations ]
#     print('esto largo stop_word', mytokens)
    
    return mytokens