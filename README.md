# ASR Benchmark

Browser-based speech-to-text benchmark for comparing multiple transcription models from the same recorded or uploaded WAV turn.

**Live app:** https://szhaomsft.github.io/ASRBenchmark/  
**Repository:** https://github.com/szhaomsft/ASRBenchmark

## Features

- Record microphone input in the browser.
- Upload an existing `.wav` file.
- Convert audio to 16 kHz mono WAV before transcription.
- Run supported models in parallel and show results side by side.
- Track per-model latency with average, P50, and P95 summaries.
- Export each turn as WAV.
- Export the full benchmark list as a ZIP containing each turn's WAV, JSON result, and text summary.

## Supported models

- Azure Speech Fast Transcription
- Azure Speech MAI Transcribe 1.5
- Azure Speech LLM Speech transcription task
- ElevenLabs Speech-to-Text Scribe v2

## Credentials

The app is static and does not include any default API keys.

Keys entered in the UI are stored only in the current browser's `localStorage`:

- Azure Speech key and region
- ElevenLabs API key

They are not committed to the repository and are not stored on a server by this app.

## Local development

```powershell
npm install
npm run dev
```

Open the local Vite URL shown in the terminal.

## Build

```powershell
npm run build
```

## Deployment

The app deploys to GitHub Pages from `main` using `.github/workflows/deploy.yml`.

The production build receives the deployed commit SHA through `VITE_GIT_COMMIT`, and the app displays links to the repository and exact deployed commit in the header.
