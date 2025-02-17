import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator

# Load the pre-trained model
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

# Load the mixture file
mixture_file = "C:/Users/Hp/21k4500/7thSemester/LIP/frameextract/Voice_Separation/output/comb1/x7g62-asevw.wav"

# Perform separation directly from the file
est_sources = model.separate_file(path=mixture_file)

# Check how many sources were separated 
num_sources = est_sources.shape[2]
print(f"Number of separated sources: {num_sources}")

# Save the separated sources conditionally
if num_sources > 0:
    torchaudio.save("output_tl_a.wav", est_sources[:, :, 0].detach().cpu(), 8000)
if num_sources > 1:
    torchaudio.save("output_tl_b.wav", est_sources[:, :, 1].detach().cpu(), 8000)
else:
    print("Only one source was separated. No second source to save.")
