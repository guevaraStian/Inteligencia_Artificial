
# El siguiente software sirve para detectar un idioma en un audio
# Se ingresa 1 audio en archivos archivos y se detecta el idioma usando IA
# Usando la librerias transformers, torch y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install transformers torch

import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DetectorAudioIdioma:

    def __init__(self):
        print("Cargando modelo Whisper para obtener texto de lo dicho en el audio...")
        self.asr_model = whisper.load_model("small")  # puedes usar tiny/base/small/medium/large
        print("El Modelo Whisper se cargado")

        print("\nSe empieza a entrenar el modelo solicitado.")
        self.tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.lang_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.labels = self.lang_model.config.id2label
        print("Modelo de inteligencia artificial, procesado")

    def Audio_a_texto(self, archivo_wav: str) -> str:
        print(f"\nPasando el audio al texto '{archivo_wav}' ...")
        result = self.asr_model.transcribe(archivo_wav, fp16=False)
        texto = result["text"].strip()
        print("Se paso de audio a texto, sin problema")
        return texto

    def detectar_idioma(self, texto: str) -> str:
        inputs = self.tokenizer(texto, return_tensors="pt")
        with torch.no_grad():
            outputs = self.lang_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return self.labels[pred]


if __name__ == "__main__":

    print("=== AUDIO → TEXTO → IDIOMA ===\n")

    archivo = input("Ingrese el nombre del archivo WAV (ejemplo: audio.wav): ")

    detector = DetectorAudioIdioma()

    # 1. Transcribir
    texto = detector.Audio_a_texto(archivo)
    print(f"\nLas palabras que se escuchan en el audio son :\n{texto}")

    # 2. Detectar idioma
    idioma = detector.detectar_idioma(texto)
    print(f"\nEl idioma que se detecto es : {idioma}\n")