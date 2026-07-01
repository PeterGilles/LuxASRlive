# LuxASRlive
LuxASR - experimental live ASR and translation

# Overview
The gateway service is the real-time, browser-facing front door for live transcription: it accepts
audio (from microphone and other inline sources) over a WebSocket connection, performs voice
activity detection, buffers the speech and forwards it to the backend ASR API. It then streams
transcription (and optional translation) back to the client.
