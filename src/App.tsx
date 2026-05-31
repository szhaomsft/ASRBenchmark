import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import JSZip from 'jszip';

type ModelId = 'fast' | 'mai' | 'llm' | 'elevenlabs';
type ModelState = 'idle' | 'processing' | 'completed' | 'error';

interface Settings {
  apiKey: string;
  elevenLabsApiKey: string;
  region: string;
  language: string;
}

interface TranscriptSegment {
  text: string;
  offset: number;
  duration: number;
  confidence: number;
  speaker?: string | number;
  locale?: string;
}

interface TranscriptResult {
  fullText: string;
  segments: TranscriptSegment[];
  language: string;
  duration: number;
}

interface ModelResult {
  state: ModelState;
  transcript?: TranscriptResult;
  error?: string;
  elapsedMs?: number;
}

interface Turn {
  id: string;
  recordedAt: number;
  sourceName: string;
  audioBlob: Blob;
  audioUrl: string;
  durationSeconds: number;
  sizeBytes: number;
  results: Record<ModelId, ModelResult>;
}

interface SpeechModel {
  id: ModelId;
  name: string;
  description: string;
}

interface WavConversion {
  audioBlob: Blob;
  durationSeconds: number;
}

interface LatencyStats {
  model: SpeechModel;
  count: number;
  averageMs: number;
  p50Ms: number;
  p95Ms: number;
}

const MODELS: SpeechModel[] = [
  {
    id: 'fast',
    name: 'Fast Transcription',
    description: 'Azure Speech fast batch endpoint',
  },
  {
    id: 'mai',
    name: 'MAI Transcribe',
    description: 'MAI-Transcribe-1.5',
  },
  {
    id: 'llm',
    name: 'LLM Speech',
    description: 'Enhanced transcription task',
  },
  {
    id: 'elevenlabs',
    name: 'ElevenLabs STT',
    description: 'Scribe v2',
  },
];

const MAI_TRANSCRIBE_MODEL = 'mai-transcribe-1.5';
const ELEVENLABS_TRANSCRIBE_MODEL = 'scribe_v2';

const DEFAULT_RESULTS: Record<ModelId, ModelResult> = {
  fast: { state: 'idle' },
  mai: { state: 'idle' },
  llm: { state: 'idle' },
  elevenlabs: { state: 'idle' },
};

const FAST_TRANSCRIPTION_LANGUAGES = [
  'en-US',
  'zh-CN',
  'de-DE',
  'fr-FR',
  'it-IT',
  'ja-JP',
  'es-ES',
  'pt-BR',
  'ko-KR',
  'ar-SA',
  'hi-IN',
];

const STORAGE_KEY = 'asr-benchmark-settings';
const WARMUP_TIMEOUT_MS = 1000;
const WARMUP_INTERVAL_MS = 15000;

function App() {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingError, setRecordingError] = useState('');
  const [warmupText, setWarmupText] = useState('Connection warm-up has not started.');
  const [isExportingZip, setIsExportingZip] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef<number>(0);
  const turnAudioUrlsRef = useRef<string[]>([]);

  const isConfigured = hasAnyModelConfiguration(settings);
  const isBusy = isRecording || turns.some((turn) =>
    MODELS.some((model) => turn.results[model.id].state === 'processing'),
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    const region = settings.region.trim();
    if (!region && !settings.elevenLabsApiKey.trim()) {
      setWarmupText('Enter model credentials to warm connections.');
      return;
    }

    let cancelled = false;
    if (region) {
      addPreconnectHint(getSpeechOrigin(region));
    }
    addPreconnectHint(getElevenLabsOrigin());
    setWarmupText('Warming model connections...');

    const warmAndReport = async () => {
      const warmed = await warmUpAllModelPipes(region);
      if (!cancelled) {
        setWarmupText(warmed ? 'Model connections are warm.' : 'Warm-up attempted; browser will retry shortly.');
      }
    };

    void warmAndReport();

    const intervalId = window.setInterval(() => {
      void warmAndReport();
    }, WARMUP_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [settings.region, settings.elevenLabsApiKey]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }
      turnAudioUrlsRef.current.forEach((audioUrl) => URL.revokeObjectURL(audioUrl));
    };
  }, []);

  const updateSetting = (key: keyof Settings, value: string) => {
    setSettings((current) => {
      const nextSettings = { ...current, [key]: value };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSettings));
      return nextSettings;
    });
  };

  const startRecording = useCallback(async () => {
    if (!isConfigured) {
      setRecordingError('Enter Azure Speech or ElevenLabs credentials before recording.');
      return;
    }

    try {
      setRecordingError('');
      void warmUpAllModelPipes(settings.region);
      chunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: getSupportedMimeType(),
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        void handleRecordingComplete(settings, Date.now() - recordingStartedAtRef.current);
      };

      mediaRecorder.start(100);
      recordingStartedAtRef.current = Date.now();
      setRecordingSeconds(0);
      setIsRecording(true);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - recordingStartedAtRef.current) / 1000));
      }, 250);
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : 'Unable to start recording.');
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setIsRecording(false);
    }
  }, [isConfigured, settings]);

  const stopRecording = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
    }
    setIsRecording(false);
  }, []);

  const handleRecordButtonClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  };

  const handleRecordingComplete = async (snapshotSettings: Settings, elapsedMs: number) => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    try {
      const sourceBlob = new Blob(chunksRef.current, { type: mediaRecorderRef.current?.mimeType || 'audio/webm' });
      if (sourceBlob.size === 0) {
        throw new Error('No audio was recorded.');
      }

      const conversion = await convertBlobToWav16k(sourceBlob);
      addTurnAndRun(conversion.audioBlob, Math.max(conversion.durationSeconds, elapsedMs / 1000), 'Recorded audio', snapshotSettings);
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : 'Failed to process recording.');
    }
  };

  const handleWaveUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file) {
      return;
    }

    if (!isConfigured) {
      setRecordingError('Enter Azure Speech or ElevenLabs credentials before uploading audio.');
      return;
    }

    if (!isWaveFile(file)) {
      setRecordingError('Upload a .wav file.');
      return;
    }

    try {
      setRecordingError('');
      const conversion = await convertBlobToWav16k(file);
      addTurnAndRun(conversion.audioBlob, conversion.durationSeconds, file.name, settings);
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : 'Failed to process uploaded WAV.');
    }
  };

  const addTurnAndRun = (audioBlob: Blob, durationSeconds: number, sourceName: string, snapshotSettings: Settings) => {
    const turnId = crypto.randomUUID();
    const audioUrl = URL.createObjectURL(audioBlob);
    turnAudioUrlsRef.current.push(audioUrl);

    const turn: Turn = {
      id: turnId,
      recordedAt: Date.now(),
      sourceName,
      audioBlob,
      audioUrl,
      durationSeconds: Math.max(1, Math.round(durationSeconds)),
      sizeBytes: audioBlob.size,
      results: cloneDefaultResults('processing'),
    };

    setTurns((current) => [turn, ...current]);
    runAllTranscriptions(turnId, audioBlob, snapshotSettings);
  };

  const runAllTranscriptions = (turnId: string, audioBlob: Blob, snapshotSettings: Settings) => {
    MODELS.forEach((model) => {
      const startedAt = performance.now();
      transcribeWithModel(model.id, audioBlob, snapshotSettings)
        .then((transcript) => {
          updateTurnResult(turnId, model.id, {
            state: 'completed',
            transcript,
            elapsedMs: Math.round(performance.now() - startedAt),
          });
        })
        .catch((error: unknown) => {
          updateTurnResult(turnId, model.id, {
            state: 'error',
            error: error instanceof Error ? error.message : 'Transcription failed.',
            elapsedMs: Math.round(performance.now() - startedAt),
          });
        });
    });
  };

  const updateTurnResult = (turnId: string, modelId: ModelId, result: ModelResult) => {
    setTurns((current) =>
      current.map((turn) =>
        turn.id === turnId
          ? {
              ...turn,
              results: {
                ...turn.results,
                [modelId]: result,
              },
            }
          : turn,
      ),
    );
  };

  const clearTurns = () => {
    turns.forEach((turn) => URL.revokeObjectURL(turn.audioUrl));
    turnAudioUrlsRef.current = [];
    setTurns([]);
  };

  const exportTurnsZip = async () => {
    if (turns.length === 0 || isExportingZip) {
      return;
    }

    try {
      setIsExportingZip(true);
      setRecordingError('');
      const zip = new JSZip();
      const orderedTurns = [...turns].reverse();
      const manifest = {
        exportedAt: new Date().toISOString(),
        turnCount: orderedTurns.length,
        models: MODELS.map((model) => ({
          id: model.id,
          name: model.name,
          description: model.description,
        })),
        turns: orderedTurns.map((turn, index) => buildTurnExportSummary(turn, index + 1)),
      };

      zip.file('manifest.json', JSON.stringify(manifest, null, 2));

      orderedTurns.forEach((turn, index) => {
        const turnNumber = index + 1;
        const folderName = `turn-${String(turnNumber).padStart(3, '0')}-${sanitizeFilePart(turn.sourceName)}`;
        const folder = zip.folder(folderName);
        if (!folder) {
          throw new Error(`Unable to create ZIP folder for turn ${turnNumber}.`);
        }

        const summary = buildTurnExportSummary(turn, turnNumber);
        folder.file('audio.wav', turn.audioBlob);
        folder.file('result.json', JSON.stringify(summary, null, 2));
        folder.file('result.txt', formatTurnResultText(summary));
      });

      const zipBlob = await zip.generateAsync({ type: 'blob' });
      downloadBlob(zipBlob, `asr-benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}.zip`);
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : 'Failed to export ZIP.');
    } finally {
      setIsExportingZip(false);
    }
  };

  const newestTurnStatus = useMemo(() => {
    const latest = turns[0];
    if (!latest) {
      return 'No recordings yet.';
    }

    const completed = MODELS.filter((model) => latest.results[model.id].state === 'completed').length;
    const failed = MODELS.filter((model) => latest.results[model.id].state === 'error').length;
    const processing = MODELS.length - completed - failed;
    return processing > 0
      ? `${processing} model${processing === 1 ? '' : 's'} still processing.`
      : `Completed ${completed}, failed ${failed}.`;
  }, [turns]);

  const latencyStats = useMemo(() => calculateLatencyStats(turns), [turns]);

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Azure Speech ASR benchmark</p>
          <h1>Record once, compare three transcribers side by side.</h1>
          <p className="subtitle">
            The page records microphone input, converts the turn to 16 kHz WAV, sends it to Fast
            Transcription, MAI Transcribe, LLM Speech, and ElevenLabs STT in parallel, then keeps each turn in a list.
            Model pipes are pre-warmed and kept active to reduce first-request latency.
          </p>
        </div>
        <div className="hero-card">
          <button
            className={isRecording ? 'record-button stop' : 'record-button'}
            onClick={handleRecordButtonClick}
            disabled={!isRecording && !isConfigured}
          >
            <span className="record-dot" />
            {isRecording ? 'Stop and transcribe' : 'Record turn'}
          </button>
          <label className={isConfigured && !isRecording ? 'upload-button' : 'upload-button disabled'}>
            Upload WAV
            <input
              type="file"
              accept=".wav,audio/wav,audio/x-wav,wav"
              onChange={handleWaveUpload}
              disabled={!isConfigured || isRecording}
            />
          </label>
          <div className="timer">{formatSeconds(recordingSeconds)}</div>
          <p>{newestTurnStatus}</p>
          <p className="warmup-status">{warmupText}</p>
        </div>
      </section>

      <section className="settings-card">
        <label>
          Azure Speech key
          <input
            type="password"
            value={settings.apiKey}
            onChange={(event) => updateSetting('apiKey', event.target.value)}
            placeholder="Paste subscription key"
            autoComplete="new-password"
            spellCheck={false}
          />
        </label>
        <label>
          ElevenLabs key
          <input
            type="password"
            value={settings.elevenLabsApiKey}
            onChange={(event) => updateSetting('elevenLabsApiKey', event.target.value)}
            placeholder="Paste ElevenLabs key"
            autoComplete="new-password"
            spellCheck={false}
          />
        </label>
        <label>
          Region
          <input
            value={settings.region}
            onChange={(event) => updateSetting('region', event.target.value)}
            placeholder="eastus"
          />
        </label>
        <label>
          Language
          <select value={settings.language} onChange={(event) => updateSetting('language', event.target.value)}>
            <option value="auto">Auto detect</option>
            {FAST_TRANSCRIPTION_LANGUAGES.map((language) => (
              <option key={language} value={language}>
                {language}
              </option>
            ))}
          </select>
        </label>
      </section>

      {recordingError && <div className="error-banner">{recordingError}</div>}

      <section className="turns-header">
        <div>
          <h2>Turns</h2>
          <p>Each row keeps the captured WAV and all three model outputs.</p>
        </div>
        <div className="turn-actions">
          <button className="secondary-button" onClick={exportTurnsZip} disabled={turns.length === 0 || isExportingZip}>
            {isExportingZip ? 'Exporting...' : 'Export ZIP'}
          </button>
          <button className="secondary-button" onClick={clearTurns} disabled={turns.length === 0 || isBusy || isExportingZip}>
            Clear list
          </button>
        </div>
      </section>

      <LatencySummary stats={latencyStats} />

      <section className="turn-list">
        {turns.length === 0 ? (
          <div className="empty-state">Record a turn to start comparing transcription results.</div>
        ) : (
          turns.map((turn, index) => <TurnCard key={turn.id} turn={turn} turnNumber={turns.length - index} />)
        )}
      </section>
    </main>
  );
}

function TurnCard({ turn, turnNumber }: { turn: Turn; turnNumber: number }) {
  const fileName = getTurnWavFileName(turn, turnNumber);

  return (
    <article className="turn-card">
      <div className="turn-meta">
        <div>
          <h3>Turn {turnNumber}</h3>
          <p>
            {turn.sourceName} · {new Date(turn.recordedAt).toLocaleString()} ·{' '}
            {formatSeconds(turn.durationSeconds)} · {formatBytes(turn.sizeBytes)}
          </p>
        </div>
        <a
          className="export-button"
          href={turn.audioUrl}
          download={fileName}
          onClick={(event) => {
            event.preventDefault();
            downloadBlob(turn.audioBlob, fileName);
          }}
        >
          Export WAV
        </a>
      </div>
      <audio src={turn.audioUrl} controls controlsList="nodownload" />
      <div className="result-grid">
        {MODELS.map((model) => (
          <ResultColumn key={model.id} model={model} result={turn.results[model.id]} />
        ))}
      </div>
    </article>
  );
}

function ResultColumn({ model, result }: { model: SpeechModel; result: ModelResult }) {
  return (
    <section className="result-column">
      <div className="result-title">
        <div>
          <h4>{model.name}</h4>
          <p>{model.description}</p>
        </div>
        <span className={`badge ${result.state}`}>{result.state}</span>
      </div>
      {result.elapsedMs !== undefined && <div className="latency">{formatMs(result.elapsedMs)}</div>}
      {result.state === 'processing' && <div className="skeleton">Transcribing...</div>}
      {result.state === 'error' && <pre className="error-text">{result.error}</pre>}
      {result.state === 'completed' && result.transcript && (
        <>
          <p className="full-text">{result.transcript.fullText || '(No text returned)'}</p>
          <div className="segments">
            {result.transcript.segments.length === 0 ? (
              <p className="muted">No segments returned.</p>
            ) : (
              result.transcript.segments.map((segment, index) => (
                <div className="segment" key={`${segment.offset}-${index}`}>
                  <span>{formatTimestamp(segment.offset)}</span>
                  <p>{segment.text}</p>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </section>
  );
}

function LatencySummary({ stats }: { stats: LatencyStats[] }) {
  return (
    <section className="latency-summary" aria-label="Latency summary">
      {stats.map((stat) => (
        <article className="latency-card" key={stat.model.id}>
          <div>
            <h3>{stat.model.name}</h3>
            <p>{stat.count} completed sample{stat.count === 1 ? '' : 's'}</p>
          </div>
          <div className="latency-metrics">
            <Metric label="Avg" value={stat.count > 0 ? formatMs(stat.averageMs) : '--'} />
            <Metric label="P50" value={stat.count > 0 ? formatMs(stat.p50Ms) : '--'} />
            <Metric label="P95" value={stat.count > 0 ? formatMs(stat.p95Ms) : '--'} />
          </div>
        </article>
      ))}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

async function transcribeWithModel(modelId: ModelId, audioBlob: Blob, settings: Settings): Promise<TranscriptResult> {
  if (modelId === 'elevenlabs') {
    return transcribeWithElevenLabs(audioBlob, settings);
  }

  const region = settings.region.trim();
  const apiKey = settings.apiKey.trim();
  if (!region || !apiKey) {
    throw new Error('Enter the Azure Speech key and region to run this model.');
  }

  const language = settings.language;
  const definition = buildDefinition(modelId, language, true);
  const endpoint = getTranscriptionEndpoint(region, modelId);

  try {
    const response = await postTranscription(endpoint, apiKey, audioBlob, definition);
    return parseTranscriptResult(response, language);
  } catch (error) {
    if (modelId !== 'mai' || !isEnhancedModelUnsupportedError(error)) {
      throw error;
    }

    const fallbackResponse = await postTranscription(endpoint, apiKey, audioBlob, buildDefinition(modelId, language, false));
    return parseTranscriptResult(fallbackResponse, language);
  }
}

async function transcribeWithElevenLabs(audioBlob: Blob, settings: Settings): Promise<TranscriptResult> {
  const apiKey = settings.elevenLabsApiKey.trim();
  if (!apiKey) {
    throw new Error('Enter the ElevenLabs API key to run this model.');
  }

  const formData = new FormData();
  formData.append('model_id', ELEVENLABS_TRANSCRIBE_MODEL);
  formData.append('file', audioBlob, 'audio.wav');
  formData.append('tag_audio_events', 'true');
  formData.append('timestamps_granularity', 'word');
  formData.append('diarize', 'true');
  formData.append('file_format', 'other');

  const languageCode = getElevenLabsLanguageCode(settings.language);
  if (languageCode) {
    formData.append('language_code', languageCode);
  }

  const response = await fetch(`${getElevenLabsOrigin()}/v1/speech-to-text`, {
    method: 'POST',
    headers: {
      'xi-api-key': apiKey,
    },
    body: formData,
  });

  if (!response.ok) {
    const responseText = await response.text();
    throw new Error(`ElevenLabs request failed (${response.status}): ${responseText}`);
  }

  return parseElevenLabsTranscript(await response.json(), settings.language);
}

function getTranscriptionEndpoint(region: string, modelId: ModelId): string {
  const apiVersion = modelId === 'fast' ? '2024-11-15' : '2025-10-15';
  return `${getSpeechOrigin(region)}/speechtotext/transcriptions:transcribe?api-version=${apiVersion}`;
}

function getSpeechOrigin(region: string): string {
  return `https://${region}.api.cognitive.microsoft.com`;
}

function getElevenLabsOrigin(): string {
  return 'https://api.elevenlabs.io';
}

function addPreconnectHint(origin: string): void {
  const existing = document.querySelector(`link[rel="preconnect"][href="${origin}"]`);
  if (existing) {
    return;
  }

  const preconnect = document.createElement('link');
  preconnect.rel = 'preconnect';
  preconnect.href = origin;
  preconnect.crossOrigin = 'anonymous';
  document.head.append(preconnect);

  const dnsPrefetch = document.createElement('link');
  dnsPrefetch.rel = 'dns-prefetch';
  dnsPrefetch.href = origin;
  document.head.append(dnsPrefetch);
}

async function warmUpModelPipe(modelId: ModelId, region: string): Promise<boolean> {
  const origin = modelId === 'elevenlabs' ? getElevenLabsOrigin() : getSpeechOrigin(region.trim());
  if (modelId !== 'elevenlabs' && !region.trim()) {
    return false;
  }

  const endpoint = modelId === 'elevenlabs'
    ? `${origin}/v1/speech-to-text`
    : getTranscriptionEndpoint(region.trim(), modelId);
  addPreconnectHint(origin);

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), WARMUP_TIMEOUT_MS);

  try {
    await fetch(endpoint, {
      method: 'HEAD',
      mode: 'no-cors',
      cache: 'no-store',
      signal: controller.signal,
    });
    return true;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function warmUpAllModelPipes(region: string): Promise<boolean> {
  const results = await Promise.all(MODELS.map((model) => warmUpModelPipe(model.id, region)));
  return results.some(Boolean);
}

function buildDefinition(modelId: ModelId, language: string, includeMaiModel: boolean): Record<string, unknown> {
  if (modelId === 'fast') {
    return {
      locales: language === 'auto' ? FAST_TRANSCRIPTION_LANGUAGES : [language],
    };
  }

  const definition: Record<string, unknown> = {
    enhancedMode: modelId === 'mai' && includeMaiModel
      ? { enabled: true, model: MAI_TRANSCRIBE_MODEL }
      : { enabled: true, task: 'transcribe' },
  };

  if (language !== 'auto') {
    definition.locales = [language.split('-')[0].toLowerCase()];
  }

  return definition;
}

function getElevenLabsLanguageCode(language: string): string | undefined {
  return language === 'auto' ? undefined : language.split('-')[0].toLowerCase();
}

async function postTranscription(
  endpoint: string,
  apiKey: string,
  audioBlob: Blob,
  definition: Record<string, unknown>,
): Promise<unknown> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'audio.wav');
  formData.append('definition', JSON.stringify(definition));

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Ocp-Apim-Subscription-Key': apiKey,
    },
    body: formData,
  });

  if (!response.ok) {
    const responseText = await response.text();
    let code = '';
    try {
      const parsed = JSON.parse(responseText);
      if (isRecord(parsed) && typeof parsed.code === 'string') {
        code = parsed.code;
      }
    } catch {
      // The error body is often plain text; preserve it below.
    }

    const error = new Error(`API request failed (${response.status}): ${responseText}`);
    Object.assign(error, { status: response.status, code, responseText });
    throw error;
  }

  return response.json();
}

function isEnhancedModelUnsupportedError(error: unknown): boolean {
  if (!(error instanceof Error) || !('status' in error) || !('code' in error) || !('responseText' in error)) {
    return false;
  }

  return error.status === 400 &&
    error.code === 'InvalidRequest' &&
    typeof error.responseText === 'string' &&
    error.responseText.includes('Enhanced mode with model is currently not supported yet');
}

function parseTranscriptResult(apiResponse: unknown, language: string): TranscriptResult {
  const response = isRecord(apiResponse) ? apiResponse : {};
  const combinedPhrases = Array.isArray(response.combinedPhrases) ? response.combinedPhrases : [];
  const combinedRecognizedPhrases = Array.isArray(response.combinedRecognizedPhrases)
    ? response.combinedRecognizedPhrases
    : [];
  const phrases = Array.isArray(response.phrases) ? response.phrases : [];
  const recognizedPhrases = Array.isArray(response.recognizedPhrases) ? response.recognizedPhrases : [];

  const fullText = getCombinedText(combinedPhrases, combinedRecognizedPhrases);
  const segments = phrases.length > 0
    ? parseModernSegments(phrases)
    : parseLegacySegments(recognizedPhrases);
  const duration = segments.length > 0
    ? Math.max(...segments.map((segment) => segment.offset + segment.duration))
    : 0;

  return {
    fullText,
    segments,
    language,
    duration,
  };
}

function getCombinedText(combinedPhrases: unknown[], combinedRecognizedPhrases: unknown[]): string {
  const phrase = combinedPhrases.find(isRecord);
  if (phrase && typeof phrase.text === 'string') {
    return phrase.text;
  }

  const recognizedPhrase = combinedRecognizedPhrases.find(isRecord);
  if (recognizedPhrase && typeof recognizedPhrase.display === 'string') {
    return recognizedPhrase.display;
  }

  return '';
}

function parseModernSegments(phrases: unknown[]): TranscriptSegment[] {
  return phrases
    .filter(isRecord)
    .map((phrase) => {
      const nBest = Array.isArray(phrase.nBest) && isRecord(phrase.nBest[0]) ? phrase.nBest[0] : undefined;
      const confidence = getNumber(phrase.confidence, getNumber(nBest?.confidence, 0));
      const text = getString(phrase.text, getString(nBest?.display, ''));
      return {
        text,
        offset: getNumber(phrase.offsetMilliseconds, parseTimestamp(phrase.offset ?? phrase.offsetInTicks)),
        duration: getNumber(phrase.durationMilliseconds, parseTimestamp(phrase.duration ?? phrase.durationInTicks)),
        confidence,
        speaker: typeof phrase.speaker === 'string' || typeof phrase.speaker === 'number' ? phrase.speaker : undefined,
        locale: typeof phrase.locale === 'string' ? phrase.locale : undefined,
      };
    })
    .filter((segment) => segment.text.length > 0);
}

function parseLegacySegments(phrases: unknown[]): TranscriptSegment[] {
  return phrases
    .filter(isRecord)
    .map((phrase) => {
      const nBest = Array.isArray(phrase.nBest) && isRecord(phrase.nBest[0]) ? phrase.nBest[0] : undefined;
      if (!nBest) {
        return undefined;
      }

      return {
        text: getString(nBest.display, ''),
        offset: parseTimestamp(phrase.offset),
        duration: parseTimestamp(phrase.duration),
        confidence: getNumber(nBest.confidence, 0),
      };
    })
    .filter((segment): segment is TranscriptSegment => Boolean(segment?.text));
}

function parseElevenLabsTranscript(apiResponse: unknown, requestedLanguage: string): TranscriptResult {
  const response = isRecord(apiResponse) ? apiResponse : {};
  const chunks = Array.isArray(response.transcripts) ? response.transcripts.filter(isRecord) : [response];
  const transcriptChunks = chunks.filter(isRecord);
  const fullText = transcriptChunks
    .map((chunk) => getString(chunk.text, ''))
    .filter(Boolean)
    .join('\n');
  const segments = transcriptChunks.flatMap(parseElevenLabsSegments);
  const durationSeconds = getNumber(response.audio_duration_secs, 0);
  const detectedLanguage = getString(transcriptChunks.find((chunk) => typeof chunk.language_code === 'string')?.language_code, requestedLanguage);
  const duration = segments.length > 0
    ? Math.max(...segments.map((segment) => segment.offset + segment.duration))
    : durationSeconds * 1000;

  return {
    fullText,
    segments,
    language: detectedLanguage,
    duration,
  };
}

function parseElevenLabsSegments(chunk: Record<string, unknown>): TranscriptSegment[] {
  const words = Array.isArray(chunk.words) ? chunk.words.filter(isRecord) : [];
  return words
    .map((word) => {
      const startSeconds = getNullableNumber(word.start);
      const endSeconds = getNullableNumber(word.end);
      const logprob = getNullableNumber(word.logprob);
      const speaker = typeof word.speaker_id === 'string' ? word.speaker_id : undefined;
      const offset = startSeconds === undefined ? 0 : startSeconds * 1000;
      const duration = startSeconds === undefined || endSeconds === undefined
        ? 0
        : Math.max(0, (endSeconds - startSeconds) * 1000);

      return {
        text: getString(word.text, ''),
        offset,
        duration,
        confidence: logprob === undefined ? 0 : Math.max(0, Math.min(1, Math.exp(logprob))),
        speaker,
        locale: typeof chunk.language_code === 'string' ? chunk.language_code : undefined,
      };
    })
    .filter((segment) => segment.text.trim().length > 0);
}

function buildTurnExportSummary(turn: Turn, turnNumber: number) {
  return {
    turnNumber,
    id: turn.id,
    sourceName: turn.sourceName,
    recordedAt: new Date(turn.recordedAt).toISOString(),
    durationSeconds: turn.durationSeconds,
    sizeBytes: turn.sizeBytes,
    audioFile: 'audio.wav',
    results: Object.fromEntries(
      MODELS.map((model) => {
        const result = turn.results[model.id];
        return [
          model.id,
          {
            modelName: model.name,
            state: result.state,
            latencyMs: result.elapsedMs ?? null,
            error: result.error ?? null,
            fullText: result.transcript?.fullText ?? '',
            language: result.transcript?.language ?? null,
            durationMs: result.transcript?.duration ?? null,
            segments: result.transcript?.segments ?? [],
          },
        ];
      }),
    ),
  };
}

function formatTurnResultText(summary: ReturnType<typeof buildTurnExportSummary>): string {
  const lines = [
    `Turn ${summary.turnNumber}`,
    `Source: ${summary.sourceName}`,
    `Recorded: ${summary.recordedAt}`,
    `Audio: ${summary.audioFile}`,
    `Duration: ${summary.durationSeconds}s`,
    `Size: ${summary.sizeBytes} bytes`,
    '',
  ];

  for (const model of MODELS) {
    const result = summary.results[model.id];
    lines.push(`## ${result.modelName}`);
    lines.push(`State: ${result.state}`);
    lines.push(`Latency: ${result.latencyMs === null ? 'n/a' : `${result.latencyMs}ms`}`);
    if (result.error) {
      lines.push(`Error: ${result.error}`);
    }
    lines.push('Text:');
    lines.push(result.fullText || '(empty)');
    lines.push('');
  }

  return `${lines.join('\n')}\n`;
}

function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function sanitizeFilePart(value: string): string {
  const cleaned = value
    .replace(/\.[^/.]+$/, '')
    .replace(/[^a-z0-9-_]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
  return cleaned || 'audio';
}

function getTurnWavFileName(turn: Turn, turnNumber: number): string {
  const timestamp = new Date(turn.recordedAt).toISOString().replace(/[:.]/g, '-');
  return `turn-${String(turnNumber).padStart(3, '0')}-${timestamp}-${sanitizeFilePart(turn.sourceName)}.wav`;
}

function calculateLatencyStats(turns: Turn[]): LatencyStats[] {
  return MODELS.map((model) => {
    const latencies = turns
      .map((turn) => turn.results[model.id])
      .filter((result) => result.state === 'completed' && typeof result.elapsedMs === 'number')
      .map((result) => result.elapsedMs as number)
      .sort((a, b) => a - b);

    if (latencies.length === 0) {
      return {
        model,
        count: 0,
        averageMs: 0,
        p50Ms: 0,
        p95Ms: 0,
      };
    }

    return {
      model,
      count: latencies.length,
      averageMs: Math.round(latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length),
      p50Ms: percentile(latencies, 0.5),
      p95Ms: percentile(latencies, 0.95),
    };
  });
}

function percentile(sortedValues: number[], percentileValue: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = Math.ceil(percentileValue * sortedValues.length) - 1;
  return sortedValues[Math.max(0, Math.min(sortedValues.length - 1, index))];
}

async function convertBlobToWav16k(sourceBlob: Blob): Promise<WavConversion> {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextCtor({ sampleRate: 16000 });
  try {
    const audioBuffer = await audioContext.decodeAudioData(await sourceBlob.arrayBuffer());
    const samples = mixToMono(audioBuffer);
    const wavBuffer = encodeWav(samples, audioBuffer.sampleRate);
    return {
      audioBlob: new Blob([wavBuffer], { type: 'audio/wav' }),
      durationSeconds: audioBuffer.duration,
    };
  } finally {
    await audioContext.close();
  }
}

function isWaveFile(file: File): boolean {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return name.endsWith('.wav') || type === 'audio/wav' || type === 'audio/x-wav' || type === 'audio/wave';
}

function mixToMono(audioBuffer: AudioBuffer): Float32Array {
  const mixed = new Float32Array(audioBuffer.length);
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < data.length; i += 1) {
      mixed[i] += data[i] / audioBuffer.numberOfChannels;
    }
  }
  return mixed;
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }

  return buffer;
}

function writeString(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function getSupportedMimeType(): string {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function cloneDefaultResults(state: ModelState): Record<ModelId, ModelResult> {
  return {
    fast: { ...DEFAULT_RESULTS.fast, state },
    mai: { ...DEFAULT_RESULTS.mai, state },
    llm: { ...DEFAULT_RESULTS.llm, state },
    elevenlabs: { ...DEFAULT_RESULTS.elevenlabs, state },
  };
}

function loadSettings(): Settings {
  const fallback = {
    apiKey: '',
    elevenLabsApiKey: '',
    region: 'eastus',
    language: 'auto',
  };

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return fallback;
    }

    const parsed = JSON.parse(stored);
    if (!isRecord(parsed)) {
      return fallback;
    }

    return {
      apiKey: getString(parsed.apiKey, fallback.apiKey),
      elevenLabsApiKey: getString(parsed.elevenLabsApiKey, fallback.elevenLabsApiKey),
      region: getString(parsed.region, fallback.region),
      language: getString(parsed.language, fallback.language),
    };
  } catch {
    return fallback;
  }
}

function hasAnyModelConfiguration(settings: Settings): boolean {
  return (settings.apiKey.trim().length > 0 && settings.region.trim().length > 0) ||
    settings.elevenLabsApiKey.trim().length > 0;
}

function getNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function getNullableNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function getString(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function parseTimestamp(value: unknown): number {
  if (typeof value === 'number') {
    return value / 10000;
  }

  if (typeof value !== 'string') {
    return 0;
  }

  const match = value.match(/PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?/);
  if (!match) {
    return 0;
  }

  const hours = Number.parseFloat(match[1] || '0');
  const minutes = Number.parseFloat(match[2] || '0');
  const seconds = Number.parseFloat(match[3] || '0');
  return (hours * 3600 + minutes * 60 + seconds) * 1000;
}

function formatSeconds(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function formatTimestamp(ms: number): string {
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

export default App;
