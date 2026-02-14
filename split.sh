ffmpeg -i recordings/2026-02-12_Peacock_Token_Call.mp4 \
-map 0:a:1 -acodec pcm_s16le -ar 16000 -ac 1 recordings/self.wav \
-map 0:a:2 -acodec pcm_s16le -ar 16000 -ac 1 recordings/others.wav
