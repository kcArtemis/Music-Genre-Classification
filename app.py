import streamlit as st
import librosa.feature as lib_fea
from pydub import AudioSegment
@st.cache(allow_output_mutation=True)
def load_model():
  model='model.pkl'
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
st.write("""
         # Music Genre Classification
         """
         )
file = st.file_uploader("Please upload a music mp3 file", type=["mp3", "wav"])
import librosa
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(audio_data, model):
    y = audio_data
    feature_ext = []
    for i in range(int(np.floor(len(y)/66149))):
      data = y[i*66149:(i+1)*66149]
      chroma_stft = lib_fea.chroma_stft(y=data)
      rms = lib_fea.rms(y=data)
      spectral_centroid = lib_fea.spectral_centroid(y=data)
      spectral_bandwidth = lib_fea.spectral_bandwidth(y=data)
      roll_off = lib_fea.spectral_rolloff(y=data)
      zero_crossing_rate = lib_fea.zero_crossing_rate(y=data)
      harmony = librosa.effects.harmonic(y=data)
      perceptr = librosa.effects.percussive(y=data)
      tempo = librosa.beat.tempo(y=data)
      mfcc = lib_fea.mfcc(y=data, n_mfcc=20)
      feature_per_3_sec = [chroma_stft.mean(), chroma_stft.var(), rms.mean(), rms.var(), spectral_centroid.mean(), spectral_centroid.var(), spectral_bandwidth.mean(), spectral_bandwidth.var(), roll_off.mean(), roll_off.var(), zero_crossing_rate.mean(), zero_crossing_rate.var(), harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var(), tempo, mfcc[0].mean(), mfcc[0].var(), mfcc[1].mean(), mfcc[1].var(), mfcc[2].mean(), mfcc[2].var(), mfcc[3].mean(), mfcc[3].var(), mfcc[4].mean(), mfcc[4].var(), mfcc[5].mean(), mfcc[5].var(), mfcc[6].mean(), mfcc[6].var(), mfcc[7].mean(), mfcc[7].var(), mfcc[8].mean(), mfcc[8].var(), mfcc[9].mean(), mfcc[9].var(), mfcc[10].mean(), mfcc[10].var(), mfcc[11].mean(), mfcc[11].var(), mfcc[12].mean(), mfcc[12].var(), mfcc[13].mean(), mfcc[13].var(), mfcc[14].mean(), mfcc[14].var(), mfcc[15].mean(), mfcc[15].var(), mfcc[16].mean(), mfcc[16].var(), mfcc[17].mean(), mfcc[17].var(), mfcc[18].mean(), mfcc[18].var(), mfcc[19].mean(), mfcc[19].var()]

    feature_ext.append(feature_per_3_sec)
    prediction = model.predict(feature_ext)    
    return prediction

if file is None:
    st.text("Please upload an audio file")
else:
    # image = Image.open(file)
    input_file = file
    output_file = 'result.wav'
    sound = AudioSegement.from_mp3(input_file)
    sound.export(output_file, format="wav")
    y , sr = librosa.load(output_file)
    st.audio(y)
    predictions = import_and_predict(y, model)
    st.write(predictions)
    print(
    "This aduio most likely belongs to {} ."
    .format(predictions)
)