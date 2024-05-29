import streamlit as st
import numpy as np
from openai import OpenAI
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch

# Lazy loading for the models
@st.cache_resource
def load_blip_model():
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"), BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


processor, model = load_blip_model()


st.title('Proyecto: Descriptor de imagenes')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

if "openai_model" not in st.session_state:
  st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
  st.session_state["messages"] = []
  setup = """Tu trabajo es la de realizar descripciones detalladas de imagenes que el puede subir a la aplicacion,
  tendras a tu disposicion dos modelos para extraer informacion, hay que aclarar que las salidas estan escritas en ingles,
  por lo que al momento de entregar la descripcion completa debes de hacerlo en espagnol.

  Tras subir una imagen al sistema, el primer modelo es un modelo de descripcion de imagenes genera una descipcion generica
  de la imagen.

  Despues van a salir mensajes del modelo de deteccion de objetos con lso objetos que ha identificado, de la forma:


  ###
   Object detected: {objeto identificado}
  ###
  """
  st.session_state.messages.append({"role":"system","content":setup})

  setup = """despues de estas salidas tendras 5 oportunidades para preguntar al modelo de descripcion de imagenes,
  trabajando dentro de un ciclo loop, por lo que cada intento debe hacerce en un mensaje.

  Dicho intento debe de constar de menos de 19 caracteres, y el programa no procedera hasta que generes una entrada de menos de 19, incluyendo espacios, escrito en ingles, teniendo en cuenta que la pregunta
  debe de hacerce de forma indirecta, como se presentara en el ejemplo un poco mas adelante.

  Para obtener mas informacion puedes usar tambien los objetos detectados por el modulo de deteccion de objetos.
  Aun cuando estos no aparecen en la descripcion final


  ### Ejemplo:
Entrada--------------
Imagen de entrada: Imagen de un perro con una bicicleta atras

Salidas generadas--------------
This is a photograf of a dog with a bike

Object Detected: dog

Object Detected: bike

Object Detected: tree

Object Detected: truck

Object Detected: house

Salidas generadas por chatGPT -----------------------

This is a photo of a dog where

.....Salida del modelo de descripcion de imagenes con mas contexto

There is a truck

.....Salida del modelo de descripcion de imagenes con mas contexto

The type of dog is

.....Salida del modelo de descripcion de imagenes con mas contexto

There is a tree

.....Salida del modelo de descripcion de imagenes con mas contexto

There is a house

......Salida del modelo de descripcion de imagenes con mas contexto
--------------------------------------------------------------------
Recoleccion de datos finalizada

Salida que ve el usuario------------------------
La phot es de un perro que esta enfrete de una bicicleta de color rojo, ambos estan adentro de una casa, y en el fondo es posible observar varios pinos,
como tambien un camion estacionado.
  """
  st.session_state.messages.append({"role":"system","content":setup})

  setup = """El modelo no es capaz de responder preguntas, para obtener mas informacion de la foto, debes de buscar la posicion de los objetos, sus colores, la composicion de
  la imagen, el tipo de vestimenta de las personas, etc.
  Cuando la seccion este por empezar, en el historial del chat se indicara con

  >>> inicio de preguntas al modelo, 19 caracters maximos <<<
  Ejemplo de preguntas apra obtener informacion.
  #####
          contexto: imagen de una casa de color azul con arboles enfrente, un perro labrador cafe y un carro azul
          Argumento para preguntar por el color de la casa
          ------
          this is a house

          #####
          contexto: imagen de un desierto con un camello y una persona
          Argumento para de la persona
          -------
          There is a person in
  """
  st.session_state.messages.append({"role":"system","content":setup})

with st.sidebar:
  test = "esto es un test, suba imagen"
  with st.form("Imagen"):
    submitted = st.form_submit_button("Enviar")
    if "image" not in st.session_state:
      img_file_buffer = st.file_uploader('Upload a PNG image', type='png')
      if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
            st.session_state["image"] = img_array
            st.session_state.messages.append({"role":"assistant","content":"Imagen obtenida"})
    if "image" in st.session_state:
        st.image(st.session_state.image)
    if submitted:
      if "image" not in st.session_state:
        st.warning("No hay imagen")
      else:
            raw_image = Image.fromarray(st.session_state["image"]).convert("RGB")

            text = "this is a picture of"
            inputs = processor(raw_image, text, return_tensors="pt")

            out = model.generate(**inputs)
            st.session_state.messages.append({"role":"system","content":processor.decode(out[0], skip_special_tokens=True)})

            processor2 = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            model2 = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

            inputs = processor2(images=raw_image, return_tensors="pt")
            outputs = model2(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([raw_image.size[::-1]])
            results = processor2.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
              st.session_state.messages.append({"role":"system","content":("object detected: " + model2.config.id2label[label.item()])})
            st.session_state.messages.append({"role":"system","content":"inicio de preguntas al modelo, 19 caracters maximos"})
            for i in range(5):

                client = OpenAI(api_key=openai_api_key)

                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
                text =  response.choices[0].message.content
                if len(text) > 20:
                  text = text[:19]
                inputs = processor(raw_image, text, return_tensors="pt")

                out = model.generate(**inputs)
                st.session_state.messages.append({"role":"system","content":processor.decode(out[0], skip_special_tokens=True)})
                st.session_state.messages.append({"role":"system","content":text})

            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role":"system","content":"""Recoleccion de datos finalizada
            Debes de dar una amplia descripcion de la imagen junto los objetos encontrados y sus posiciones y o descripciones.
            A partir de este punto
            """})
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
            del st.session_state["image"]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Escribe algo"):
    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
