##Proyecto: Descriptor de Imágenes

Este proyecto utiliza Streamlit para crear una aplicación web que genera descripciones detalladas de imágenes subidas por el usuario. El sistema emplea dos modelos de aprendizaje automático: uno para la generación de descripciones de imágenes y otro para la detección de objetos.

#Descripción
La aplicación permite a los usuarios subir una imagen y obtener una descripción detallada en español. Utiliza dos modelos principales:

  1. Modelo de Descripción de Imágenes: Genera una descripción inicial de la imagen.
  2.Modelo de Detección de Objetos: Identifica y enumera los objetos presentes en la imagen.
Posteriormente, el sistema interactúa con el usuario mediante mensajes para refinar la descripción inicial, proporcionando una salida más detallada y específica.

#Uso

1. Crear tu OpenaAI key
2. Escribir tu OpenAI key en el campo indicado
3. Subir una Imagen: Sube una imagen en formato PNG a través del cargador de archivos en la barra lateral.
4.Generar Descripción: La aplicación procesará la imagen, generará una descripción inicial y detectará los objetos presentes en la imagen.
5.Refinar Descripción: La aplicación interactuará con el modelo de OpenAI para refinar la descripción de la imagen mediante una serie de preguntas.
6.Ver Resultados: La descripción final detallada se mostrará en la interfaz de usuario.
