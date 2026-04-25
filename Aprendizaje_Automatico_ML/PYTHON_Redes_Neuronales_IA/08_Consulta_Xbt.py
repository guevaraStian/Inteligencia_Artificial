# El siguiente software sirve para consultar unos datos de XBT y organizar unos datos
# Se hace una consulta a una API de bolsa y se organizan los datos
# Usando la librerias requests, json y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install requests json
# pip 25.1.1
# Python 3.13.1
import requests
import json

# Se guarda la URL de la API de XTB en una variable
Ruta_api = "https://api.xtb.com"

# Ingresamos las credenciales de API (token de acceso, etc.)
Api_token = "TU_API_KEY"

# Encabezados HTTP
Encabezados = {
    "Authorization": f"Bearer {Api_token}",
    "Content-Type": "application/json"
}

# En el siguiente codigo se Consulta datos de mercado (Ejemplo: Cotizaciones)
Respuesta = requests.get(f"{Ruta_api}/market_data", headers=Encabezados)

if Respuesta.status_code == 200:
    Datos = Respuesta.json()
    print(json.dumps(Datos, indent=2))
else:
    print(f"Error: {Respuesta.status_code}, Respuesta.text}")