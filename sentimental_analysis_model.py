import pandas as pd


#Leer los archivos de texto
file1 = pd.read_csv("sentiment_labelled_sentences/amazon_cells_labelled.txt", sep="\t", header=None)
file2 = pd.read_csv("sentiment_labelled_sentences/imdb_labelled.txt", sep="\t", header=None)
file3 = pd.read_csv("sentiment_labelled_sentences/yelp_labelled.txt", sep="\t", header=None)
#Mostrar el tamaño de cada file leido para verificar (incompletos en el file2, error de origen de la bd)
print("Número de filas en file1:", len(file1))
print("Número de filas en file2:", len(file2))
print("Número de filas en file3:", len(file3))


# Concatenar los tres archivos
combined_df = pd.concat([file1, file2, file3], ignore_index=True)
#Nombre de las columnas
combined_df.columns = ["Phrase", "tag"]
#Imprimir el dataframe
print(combined_df)

