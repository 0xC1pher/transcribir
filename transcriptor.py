import whisper
import torchaudio
import os

# Cargar el modelo de Whisper (elige 'base', 'small', 'medium', 'large' según tus necesidades)
model = whisper.load_model("small") 

def transcribe_audio(audio_path, language="en"):
    """
    Transcribe un archivo de audio utilizando el modelo Whisper.

    :param audio_path: Ruta al archivo de audio.
    :param language: Idioma de la transcripción (por defecto es inglés "en").
    :return: Transcripción del audio.
    """
    try:
        
        audio, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error al cargar el archivo de audio: {e}")
        return None

    try:
        # Si el audio no está en 16 kHz, resamplearlo
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)
    except Exception as e:
        print(f"Error durante el resampling: {e}")
        return None

    try:
        
        transcription = model.transcribe(audio, language=language)
        return transcription['text']
    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        return None


audio_file_path = "ruta_a_tu_audio.wav"
language = "es"  # Cambiar esto al idioma que desees, por ejemplo, "es" para español

if os.path.exists(audio_file_path):
    transcription = transcribe_audio(audio_file_path, language)
    if transcription:
        print("Transcripción:", transcription)
    else:
        print("No se pudo obtener la transcripción.")
else:
    print("El archivo de audio no existe.")
