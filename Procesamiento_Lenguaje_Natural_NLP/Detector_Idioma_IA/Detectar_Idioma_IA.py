
# En el siguiente codigo se muestra un
# Detección de idioma con la libreria de IA Transformers
# Primero instalamos las librerias necesarias
# pip install transformers torch


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DetectorIdiomaIA:
    def __init__(self):
        print("Empieza a cargar el modelo de IA...")
        self.tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.labels = self.model.config.id2label
        print("Finalizo la carga del modelo IA")

    def detectar(self, texto: str) -> str:
        # Se lee el texto ingresado
        inputs = self.tokenizer(texto, return_tensors="pt")

        # Se analiza su posible lenguaje
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()

        idioma = self.labels[pred_id]
        return idioma


if __name__ == "__main__":
    detector = DetectorIdiomaIA()

    print("EMPIEZA A FUNCIONAR EL DETECTOR DE IDIOMA CON IA")
    print("Por favor, Escribe un texto y presiona Enter. (Ctrl+C para salir)\n")

    while True:
        texto = input("Por favor, ingrese el texto: ")
        idioma = detector.detectar(texto)
        print(f"El idioma del texto ingresado es: {idioma}\n")