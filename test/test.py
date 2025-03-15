from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

filename = []

segments, info = model.transcribe(
    # f"/Users/mousebook/Documents/side project/2024hack/test/STT/music/4_6f1ce97a-cb5d-49a7-8ce9-38672133be17.wav",
    f"/app/music/4_6f1ce97a-cb5d-49a7-8ce9-38672133be17.wav",
    beam_size=5,
)

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

stt_result = ""

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    stt_result += segment.text

print(stt_result)
