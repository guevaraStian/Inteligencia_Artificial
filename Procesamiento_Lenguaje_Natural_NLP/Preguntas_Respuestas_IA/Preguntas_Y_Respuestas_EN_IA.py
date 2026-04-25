# El siguiente software sirve para recibir preguntas y dar respuestas en idioma ingles
# el software muestra cada respuesta con datos de inteligencia artificial
# Usando la librerias openai y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install openai
import openai

openai.api_key = "Para usar esta aplicacion hay que pedirle a chatgpt una llave ponerla aqui"

# Se guarda en una variable la pregunta completa escrita en ingles
prompt = "number of continents?"

# Se crea la variable con los datos de configuracion
response = openai.Completion.create(
    engine="text-davinci-001", prompt=prompt, max_tokens=6
)

# Muestra en pantalla la respuesta
print(response)