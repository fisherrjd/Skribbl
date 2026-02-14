import sys

import soundfile as sf
import torch
from pyannote.audio import Pipeline

audio_file_path = sys.argv[1] if len(sys.argv) > 1 else "system.wav"

# Load audio manually using soundfile
waveform, sample_rate = sf.read(audio_file_path)
# Convert to torch tensor and ensure shape is (channels, samples)
waveform = torch.from_numpy(waveform.T).float()
# If mono, add channel dimension
if waveform.dim() == 1:
    waveform = waveform.unsqueeze(0)

# Create audio dict that pyannote expects
audio_file = {"waveform": waveform, "sample_rate": sample_rate}

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# Force exactly 2 speakers since we know there are 2 people in the "others" track
diarization = pipeline(audio_file, num_speakers=2)

# Print the diarization output
# Access the speaker_diarization annotation
annotation = diarization.speaker_diarization

for segment, _, speaker in annotation.itertracks(yield_label=True):
    print(f"{segment.start:.1f}s - {segment.end:.1f}s: {speaker}")
