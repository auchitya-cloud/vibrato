from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64
import os

output_dir = "audio_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def check_key_value_lengths(input_data):
    # Extract keys and values from input data
    keys, values = extract_keys_and_values(input_data)

    # Check lengths of keys and values
    key_length = keys.shape[1]
    value_length = values.shape[1]

    if key_length != value_length:
        raise ValueError(f"Key length ({key_length}) does not match value length ({value_length})")

    # Proceed with self-attention layer

def pad_sequences(sequences, max_length):
    # Pad shorter sequences with zeros
    padded_sequences = torch.nn.functional.pad(sequences, pad=(0, max_length - sequences.shape[1]))

    return padded_sequences



@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a temporary file.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        save_path (str): path to the directory where audio should be saved.
    """

    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output"
    output_dir = "audio_output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the samples tensor has less than 2 dimensions
    if samples.dim() < 2:
        samples = samples.unsqueeze(0)  # Add an extra dimension

    # Check if the samples tensor has more than 3 dimensions
    if samples.dim() > 3:
        samples = samples.squeeze()  # Reduce the number of dimensions

    samples = samples.detach().cpu()

    # Save audio files for each sample
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

    # Return the path to the first audio file
    return audio_path



def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon= "Vibrato",
    page_title= "Vibrato Generator"
)

def main():

    st.title("Vibrato AI music generator🎵")

    with st.expander("See explanation"):
        st.write("Vibrato app built using Meta's Audiocraft library. We are using Music Gen melody model.Unleash your creativity with MusicEngi, an AI-powered music composition tool. Simply enter a text prompt or musical attributes, and Vibrato AI will generate a unique and original piece of music tailored to your specifications. Whether you're a seasoned musician or a complete beginner, Vibrato AI empowers you to explore the boundless realm of musical expression.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        print("Musci Tensors: ", music_tensors)
        save_music_file = save_audio(music_tensors)
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    

