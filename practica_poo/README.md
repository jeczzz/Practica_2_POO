# Práctica 2 de POO - Regresión lineal en C++

**Autor:** Jorge Córdoba Zota  
**Asignatura:** Lenguajes y Paradigmas de Programación - Universidad EAFIT

El programa implementa una clase de regresión lineal en C++17 con escalado de entradas y entrenamiento por descenso de gradiente. Utiliza la misma clase para estimar ventas de helados a partir de la temperatura y notas de examen a partir de cuatro variables. También mide la memoria que ocupan los modelos y los datos de sus vectores.

## Archivos del proyecto

| Archivo o carpeta | Contenido |
|---|---|
| `LinearRegression.h` y `LinearRegression.cpp` | Declaración e implementación del modelo. |
| `main.cpp` | Lectura de los CSV, entrenamiento, evaluación y análisis de memoria. |
| `pruebas.cpp` | Pruebas automáticas de la clase. |
| `datos/` | `Ice_cream_selling_data.csv` y `student_exam_scores.csv`. |
| `resultados/` | Archivos generados con las predicciones de prueba. |

## Compilación y ejecución

Se requiere un compilador compatible con C++17. El programa usa la biblioteca estándar de C++.

Los siguientes comandos se ejecutan en **PowerShell**, desde la raíz del repositorio, con `g++` disponible en la terminal.

```powershell
New-Item -ItemType Directory -Path resultados -Force | Out-Null
g++ -std=c++17 -Wall -Wextra -Wpedantic main.cpp LinearRegression.cpp -o programa.exe
.\programa.exe
```

En **Dev-C++**, se puede crear un proyecto de C++ en la raíz del repositorio, agregar `main.cpp` y `LinearRegression.cpp` y compilar con el estándar C++17. El ejecutable debe iniciarse desde esa misma carpeta para que encuentre `datos/` y `resultados/`.

## Resultados

El programa muestra los coeficientes, el sesgo, el error cuadrático medio (MSE) y las predicciones de prueba. Los CSV `helados_predicciones.csv` y `estudiantes_predicciones.csv` se guardan en `resultados/`. Cada ejecución vuelve a escribir esos archivos.

| Conjunto de datos | Muestras de entrenamiento | Muestras de prueba | MSE de prueba |
|---|---:|---:|---:|
| Helados | 39 | 10 | 89,6810 |
| Estudiantes | 160 | 40 | 5,4549 |

Al final, la consola muestra las direcciones de los modelos global, local y dinámico, el tamaño de cada objeto y el espacio que reservan sus vectores. Las direcciones pueden cambiar entre ejecuciones y los tamaños dependen de la plataforma y el compilador.

## Pruebas automáticas

```powershell
g++ -std=c++17 -Wall -Wextra -Wpedantic pruebas.cpp LinearRegression.cpp -o pruebas.exe
.\pruebas.exe
```

Las pruebas comprueban la recuperación de coeficientes conocidos, la predicción para una entrada nueva, el MSE y el rechazo de entradas inválidas. También cubren una columna constante y el reentrenamiento. Si todas pasan, la salida comienza con `OK: coeficientes conocidos`.

En Dev-C++, el proyecto de pruebas se compila con `pruebas.cpp` y `LinearRegression.cpp`. El programa principal y las pruebas se generan como ejecutables separados porque cada uno define su propia función `main()`.
