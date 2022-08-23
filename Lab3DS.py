import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop("label",axis=1)

print("Cantidad de filas y columnas:",df.shape)
print("Cantidad total de entradas",df.size)

print("Cantidad de entradas distintas de cero:",np.count_nonzero(df))

print("El porcentaje de datos distintos de cero es:",100*np.count_nonzero(df)/df.size,"%")

df.loc['media'] = df.mean()

x = df.loc['media']


digEjemplo1 = np.array(df.loc[123]).reshape(28,28)

plt.imshow(digEjemplo1)
plt.show()

digEjemplo2 = np.array(df.loc[69]).reshape(28,28)

plt.imshow(digEjemplo2)
plt.show()

digEjemplo3 = np.array(df.loc[420]).reshape(28,28)

plt.imshow(digEjemplo3)
plt.show()

digEjemplo4 = np.array(df.loc[6969]).reshape(28,28)

plt.imshow(digEjemplo4)
plt.show()

digEjemplo5 = np.array(df.loc[41999]).reshape(28,28)

plt.imshow(digEjemplo5)
plt.show()

digEjemplo6 = np.array(df.loc[1111]).reshape(28,28)

plt.imshow(digEjemplo6)
plt.show()

digEjemplo7 = np.array(df.loc[1345]).reshape(28,28)

plt.imshow(digEjemplo7)
plt.show()

digEjemplo8 = np.array(df.loc[5859]).reshape(28,28)

plt.imshow(digEjemplo8)
plt.show()

digPromedio = np.array(df.loc['media']).reshape(28,28)

plt.imshow(digPromedio)
plt.show()