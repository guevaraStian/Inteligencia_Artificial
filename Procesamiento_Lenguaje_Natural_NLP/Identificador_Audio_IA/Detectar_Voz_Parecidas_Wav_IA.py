
# El siguiente software sirve para detectar si son iguales 2 voces
# Se ingresan 2 audios en archivos diferentes y se compara las voces si son iguales
# Usando la librerias numpy, librosa y sus datos
# En el terminal de comandos importar librerias con el comando PIP 
# pip install speechbrain torch torchaudio
# pip 25.1.1
# Python 3.13.1
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa

# Cargar audio y convertir a 16 kHz
def cargar_audio(path):
    wav, sr = librosa.load(path, sr=16000)
    return wav

# Calcular similitud coseno en el audio
def similitud_coseno(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return float(np.dot(v1, v2))

# En los siguientes codigos se comparan las voces
def comparar_voces(a1, a2):
    print("\nSe carga el modelo de voz...")
    encoder = VoiceEncoder()

    print("\nSe proceso el audio 1...")
    wav1 = cargar_audio(a1)
    emb1 = encoder.embed_utterance(wav1)

    print("Se proceso el audio 2...")
    wav2 = cargar_audio(a2)
    emb2 = encoder.embed_utterance(wav2)

    print("\nEmpieza a calcular similitud ...")
    sim = similitud_coseno(emb1, emb2)

    return sim


# Main de ejecucion
if __name__ == "__main__":
    print("=== DETECTOR DE VOZ PARECIDA (IA – Resemblyzer) ===\n")

    audio1 = input("Ingrese el primer audio (WAV/MP3): ")
    audio2 = input("Ingrese el segundo audio (WAV/MP3): ")

    sim = comparar_voces(audio1, audio2)

    print(f"La similitud de voces en los 2 audios es de: {sim:.4f}")

    if sim > 0.80:
        print("Las 2 voces de los audios, son muy parecidas (probablemente la misma persona).")
    elif sim > 0.60:
        print("Las 2 voces de los audios, se parecen un poco, pero no hay certeza.")
    else:
        print("Las 2 voces de los audios, no se parecen")

    print("--------------------------------------------\n")