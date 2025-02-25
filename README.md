 Isomap (Isometric Mapping) es un algoritmo de reducción de dimensionalidad basado en la preservación de distancias geodésicas. Es útil para representar datos de alta dimensión en un espacio de menor dimensión manteniendo su estructura no lineal.
Se usa en aplicaciones como reconocimiento de patrones, análisis de datos y visualización de estructuras complejas en espacios de menor dimensión.

---

### **Implementación:**

1. **Cálculo de distancias euclidianas (`pairwise_distances`)**  
   - Se calcula la distancia euclidiana entre cada par de puntos en los datos de entrada.

2. **Construcción del grafo de vecinos más cercanos (`shortest_paths`)**  
   - Para cada punto, se conecta con sus `k` vecinos más cercanos, formando un grafo disperso.

3. **Cálculo de distancias geodésicas (`isomap`)**  
   - Se usa el algoritmo de Floyd-Warshall para encontrar la distancia más corta entre todos los pares de nodos en el grafo.

4. **Aplicación de reducción de dimensionalidad**  
   - Se usa el **centrado doble** para transformar la matriz de distancias en una matriz de Gram.
   - Se aplica **SVD (descomposición en valores singulares)** para obtener las coordenadas en el nuevo espacio de menor dimensión.

5. **Devolución de resultados**  
   - El programa devuelve las nuevas coordenadas (`Y`) y la matriz de distancias geodésicas (`G_floyd`), que pueden ser usadas para visualización.

---

**Salida en Python:**  
El código de Python que usa este módulo genera:
- Un **gráfico de dispersión** de los datos transformados.
- Un **heatmap** de la matriz de distancias geodésicas.
