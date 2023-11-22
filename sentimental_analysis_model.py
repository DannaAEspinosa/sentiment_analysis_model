import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Leer los archivos de texto
file1 = pd.read_csv("sentiment_labelled_sentences/amazon_cells_labelled.txt", sep="\t", header=None)
file2 = pd.read_csv("sentiment_labelled_sentences/imdb_labelled.txt", sep="\t", header=None)
file3 = pd.read_csv("sentiment_labelled_sentences/yelp_labelled.txt", sep="\t", header=None)

#Mostrar el tamaño de cada file leido para verificar (incompletos en el file2, error de origen de la bd)
print("Número de filas en file1:", len(file1))
print("Número de filas en file2:", len(file2))
print("Número de filas en file3:", len(file3))

# Concatenar los tres archivos
combined_df = pd.concat([file1, file2, file3], ignore_index=True)
combined_df.columns = ["Phrase", "tag"]

#Imprimir combined_df
print(combined_df)

# Función para limpiar el texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar caracteres especiales y puntuación
    return text

# Aplicar limpieza y tokenización para palabras en inglés
combined_df['Cleaned_Phrase'] = combined_df['Phrase'].apply(clean_text)
combined_df['Tokenized_Phrase'] = combined_df['Cleaned_Phrase'].apply(word_tokenize)
combined_df['Tokenized_Phrase'] = combined_df['Tokenized_Phrase'].apply(lambda x: [word.lower() for word in x])

# Descargar y eliminar palabras vacías en inglés
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
combined_df['Tokenized_Phrase'] = combined_df['Tokenized_Phrase'].apply(lambda x: [word for word in x if word not in stop_words])

# Imprimir el dataframe con las frases originales y las palabras preprocesadas en inglés
print(combined_df[['Phrase', 'Tokenized_Phrase']])
