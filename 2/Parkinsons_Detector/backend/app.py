import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from pydub import AudioSegment, silence
import noisereduce as nr
import json

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 13

# --- New function to process audio and extract features ---
def process_audio(audio_path):
    """
    Processes an audio file and extracts its features.
    
    Args:
        audio_path (str): The path to the audio file.
        
    Returns:
        A dictionary containing the mean values of the extracted features,
        or None if processing fails.
    """
    try:
        # Convert to WAV
        audio = AudioSegment.from_file(audio_path)
        wav_audio = io.BytesIO()
        audio.export(wav_audio, format="wav")
        wav_audio.seek(0)

        # Remove silence and trim to 1.5s
        audio_segment = AudioSegment.from_wav(wav_audio)
        silent_ranges = silence.detect_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
        
        trimmed_audio_segment = None
        if silent_ranges:
            first_non_silent_start = silent_ranges[0][0]
            last_non_silent_end = silent_ranges[-1][1]
            trimmed_audio_segment = audio_segment[first_non_silent_start:last_non_silent_end]
        
        if not trimmed_audio_segment or len(trimmed_audio_segment) < 1500: # 1.5s
            return {"error": "Audio is too short after trimming silence. Must be at least 1.5s."}
        
        # Trim to exactly 1.5s if it's longer
        if len(trimmed_audio_segment) > 1500:
            trimmed_audio_segment = trimmed_audio_segment[:1500]

        # Convert to numpy array and normalize
        trimmed_audio = np.array(trimmed_audio_segment.get_array_of_samples()).astype(np.float32)
        trimmed_audio /= np.max(np.abs(trimmed_audio))
        
        # Noise Reduction
        # Assume a noise part at the beginning of the audio.
        noise_clip = trimmed_audio[0:500] 
        reduced_noise_audio = nr.reduce_noise(y=trimmed_audio, sr=SAMPLE_RATE, y_noise=noise_clip, verbose=False)

        # Feature Extraction
        features = {}

        # Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=reduced_noise_audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        features["mel_spectrogram"] = np.mean(mel_spectrogram)

        # MFCC
        mfccs = librosa.feature.mfcc(y=reduced_noise_audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        features["mfcc"] = np.mean(mfccs)
        
        # Pitch (Fundamental Frequency)
        pitches, magnitudes = librosa.piptrack(y=reduced_noise_audio, sr=SAMPLE_RATE)
        features["pitch"] = np.mean(pitches)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=reduced_noise_audio, sr=SAMPLE_RATE)
        features["chroma"] = np.mean(chroma)

        # Return a dictionary of mean values
        return features

    except Exception as e:
        return {"error": str(e)}

@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get user information from the form data
    age = request.form.get('age')
    sex = request.form.get('sex')
    user_id = request.form.get('id')
    is_diagnosed = request.form.get('is_diagnosed')
    diagnosis_year = request.form.get('diagnosis_year')

    # Save the file to a temporary location
    temp_path = os.path.join("temp", user_id + ".wav")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)

    # Process the audio file
    extracted_features = process_audio(temp_path)

    # Clean up the temporary file
    os.remove(temp_path)

    if "error" in extracted_features:
        return jsonify(extracted_features), 400
    
    # Create a dictionary to hold all the data
    all_data = {
        "user_info": {
            "age": age,
            "sex": sex,
            "id": user_id,
            "is_diagnosed": is_diagnosed,
            "diagnosis_year": diagnosis_year
        },
        "features": extracted_features
    }

    # Here you would typically save the `all_data` to a database
    # For now, we will just print it to the console as a JSON string
    # to show that the data has been collected.
    print("Collected Data (as JSON):")
    print(json.dumps(all_data, indent=4))
    
    return jsonify({"features": extracted_features}), 200

if __name__ == '__main__':
    app.run(debug=True)
