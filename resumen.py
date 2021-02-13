import re
import pickle
import networkx
import numpy as np
from utils import build_feature_matrix, low_rank_svd
from gensim.summarization import summarize, keywords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from os import listdir
from os.path import isfile
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

path_resenas = "/home/link/Escritorio/PLN/aspectos/musica/"
path_to_no_sentences = "/home/link/Escritorio/PLN/aspectos/enunciados_yes"
path_to_yes_sentences = "/home/link/Escritorio/PLN/aspectos/enunciados_no"
aspectos = ('disco', 'canción', 'grupo', 'voz', 'músico', 'guitarra', 'sonido')

def ls1(path):
    return [obj for obj in listdir(path) if isfile(path + obj)]

def agrupa(path):
    files = ls1(path)
    yes_files = []
    no_files = []
    for file in files:
        if('yes' in file):
            yes_files.append(file)
        else:
            no_files.append(file)
    return yes_files, no_files

def lematiza(text, only_nouns):
    nlp = spacy.load('es')
    doc = nlp(text)
    if(only_nouns == True):
        return [word.lemma_ for word in doc if word.pos_ == 'NOUN']
    else:
        return ' '.join([word.lemma_ for word in doc])

def lee(file_names, origin):
    string_total_reviews = ""
    for file in file_names:
        f = open(origin + file, encoding='latin-1')
        contenido=""
        while(True):
            linea = f.readline()
            if linea !="" and linea !=" " and linea !="\n":
                linea= linea.replace("\n","")
                linea=linea.replace("\t","")
                linea=linea.replace("/","")
                linea=linea.replace("\x86","")
                linea=linea.replace("®","")
                linea=linea.replace("D\'","")
                contenido=contenido+linea
            if not linea:
                break
        f.close()        
        string_total_reviews = string_total_reviews + contenido
    return string_total_reviews

def resenas_enunciados(data):
    enunciado = str(sent_tokenize(data))
    lematizado = lematiza(enunciado, False)
    return sent_tokenize(lematizado)

def buscar_aspecto(enunciados, aspecto):
    seleccionados = []
    for enunciado in enunciados:
        if(aspecto in enunciado):
            seleccionados.append(enunciado)
    return seleccionados

def limpiar(a):
    cadena=a
    cadena.replace("' , '","")
    cadena.replace(", '","")
    return cadena

def cargar(path):
    f = open(path, 'rb')
    sentences = pickle.load(f)
    f.close()

    return sentences

def buscar_aspecto(sentences, aspect):
    selected_sentences = []
    for sentence in sentences:
        if(aspect in sentence):
            selected_sentences.append(sentence)
    return selected_sentences

from gensim.summarization import summarize, keywords
def text_summarization_gensim(text, summary_ratio=0.5):
    summary = summarize(text, split=True, ratio=summary_ratio)
    
    return summary

def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]    
    return sentences

def imprime(geny,genn,lsay,lsan,tery,tern,aspecto):
    G = "\033[0;32;40m"
    B = "\033[0;34;40m"
    N = "\033[0m"
    geny=limpiar(geny)
    genn=limpiar(genn)
    lasy=limpiar(lsay)
    lsan=limpiar(lsan)
    teryy=limpiar(tery)
    tern=limpiar(tern)
    print("\n")
    print(G+("*** "+aspecto+" ***")+N)
    print("\n")
    print(B+("*** "+"GENSIM"+" ***")+N)
    print("\n")
    print("*********** POSITIVO ***********")
    print(geny)
    print("\n")
    print("*********** NEGATIVO ***********")
    print(genn)
    print("\n")
    print(B+("*** "+"LSA"+" ***")+N)
    print("\n")
    print("*********** POSITIVO ***********")
    print(lsay)
    print("\n")
    print("*********** NEGATIVO ***********")
    print(lsan)
    print("\n")
    print(B+("*** "+"TEXT-RANK"+" ***")+N)
    print("\n")
    print("*********** POSITIVO ***********")
    print(tery)
    print("\n")
    print("*********** NEGATIVO ***********")
    print(tern)

yes_files, no_files = agrupa(path_resenas)
yes_string = lee(yes_files, path_resenas)
y= resenas_enunciados(yes_string)
no_string = lee(no_files, path_resenas)
n= resenas_enunciados(no_string)
#n=cargar(path_to_no_sentences)
#y = cargar(path_to_yes_sentences)
ss = " "
enunciados_no = sent_tokenize(ss.join(n))
enunciados_yes = sent_tokenize(ss.join(y))

num_sentences = 3
num_topics = 3

for aspecto in aspectos:    
    enunciados_yes_final = ' '.join(buscar_aspecto(enunciados_yes, aspecto))
    enunciados_no_final = ' '.join(buscar_aspecto(enunciados_no, aspecto))
    # Gensim
    gensim_yes = ' '.join(text_summarization_gensim(enunciados_yes_final, summary_ratio=0.2))
    gensim_no = ' '.join(text_summarization_gensim(enunciados_no_final, summary_ratio=0.2))
    # LSA
    enunciados_yes_final = parse_document(' '.join(buscar_aspecto(enunciados_yes, aspecto)))
    lsa_yes=[]
    vec, dt_matrix = build_feature_matrix(enunciados_yes_final, feature_type='frequency')
    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)
    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    sv_threshold = 0.5
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()   
    for index in top_sentence_indices:
        lsa_yes.append(enunciados_yes_final[index])
        

    enunciados_no_final = parse_document(' '.join(buscar_aspecto(enunciados_no, aspecto)))
    lsa_no = []  
    vec, dt_matrix = build_feature_matrix(enunciados_no_final, feature_type='frequency')
    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)
    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    sv_threshold = 0.5
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()    
    for index in top_sentence_indices:
        lsa_no.append(enunciados_no_final[index])
        
    lsa_no = ' '.join(lsa_no)
    lsa_yes = ' '.join(lsa_yes)
        
    
    tr_no = []
    tr_yes = []
    
    vec, dt_matrix = build_feature_matrix(enunciados_yes_final, feature_type='tfidf')
    similarity_matrix = (dt_matrix * dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()
    for index in top_sentence_indices:
        tr_yes.append(enunciados_yes_final[index])

    vec, dt_matrix = build_feature_matrix(enunciados_no_final, feature_type='tfidf')
    similarity_matrix = (dt_matrix * dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()    
    for index in top_sentence_indices:
        tr_no.append(enunciados_no_final[index])
    
    tr_no = ' '.join(tr_no)
    tr_yes = ' '.join(tr_yes)
    
    imprime(gensim_yes,gensim_no,lsa_yes,lsa_no,tr_yes,tr_no,aspecto)
