import {pipeline} from '@huggingface/transformers';

type Entity = {
  entity_group: string;
  score: number;
  word: string;
  start: number;
  end: number;
};

type ProgressEvent =
  | {status: 'initiate'; file: string; name?: string}
  | {status: 'download'; file: string; name?: string}
  | {status: 'progress'; file: string; progress?: number; loaded?: number; total?: number}
  | {status: 'done'; file: string}
  | {status: 'ready'};

const statusPill = document.querySelector('#status') as HTMLDivElement;
const statusText = statusPill.querySelector('.status-text') as HTMLSpanElement;
const textarea = document.querySelector('textarea.input-area') as HTMLTextAreaElement;
const actionButton = document.querySelector('#action') as HTMLButtonElement;
const actionLabel = actionButton.querySelector('.btn-label') as HTMLSpanElement;
const exampleButton = document.querySelector('#example') as HTMLButtonElement;
const outputArea = document.querySelector('#output') as HTMLDivElement;
const stats = document.querySelector('#stats') as HTMLSpanElement;

const downloadCard = document.querySelector('#download') as HTMLElement;
const downloadTotal = document.querySelector('#download-total') as HTMLSpanElement;
const downloadFill = document.querySelector('#download-fill') as HTMLDivElement;
const downloadFiles = document.querySelector('#download-files') as HTMLUListElement;

const EXAMPLE = `Hi team,

Please reach out to Dr. Sarah Chen at sarah.chen@medline.io or +1 (415) 555-0134 regarding the patient file. She's based at 2100 Market Street, San Francisco, CA 94114.

Backup contact: Marcus Alvarez, SSN 412-88-9031, DOB 1987-03-22.

— Jamie`;

type ClassifierFn = (text: string, options: {aggregation_strategy: string}) => Promise<Entity[]>;

const MODEL_ID = 'openai/privacy-filter';
let classifier: ClassifierFn | null = null;

function setStatus(state: 'idle' | 'loading' | 'ready' | 'busy' | 'error', text: string) {
  statusPill.dataset.state = state;
  statusText.textContent = text;
}

function setMode(mode: 'load' | 'busy' | 'redact', label: string) {
  actionButton.dataset.mode = mode;
  actionLabel.textContent = label;
}

function formatBytes(bytes: number): string {
  if (!bytes) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 10 ? 0 : 1)} ${units[i]}`;
}

type FileRow = {
  progress: number;
  loaded: number;
  total: number;
  done: boolean;
  li: HTMLLIElement;
  fill: HTMLDivElement;
  meta: HTMLSpanElement;
};

const files = new Map<string, FileRow>();

function ensureFile(name: string): FileRow {
  let row = files.get(name);
  if (row) return row;

  const li = document.createElement('li');
  li.dataset.done = 'false';
  const nameEl = document.createElement('span');
  nameEl.className = 'file-name';
  nameEl.textContent = name;
  const meta = document.createElement('span');
  meta.className = 'file-meta';
  meta.textContent = 'queued';
  const bar = document.createElement('div');
  bar.className = 'file-bar';
  const fill = document.createElement('div');
  fill.className = 'file-fill';
  bar.append(fill);
  li.append(nameEl, meta, bar);
  downloadFiles.append(li);

  row = {progress: 0, loaded: 0, total: 0, done: false, li, fill, meta};
  files.set(name, row);
  return row;
}

function updateTotals() {
  if (files.size === 0) return;
  let totalBytes = 0;
  let loadedBytes = 0;
  let allHaveTotal = true;
  for (const row of files.values()) {
    if (row.total > 0) {
      totalBytes += row.total;
      loadedBytes += row.done ? row.total : row.loaded;
    } else if (!row.done) {
      allHaveTotal = false;
    }
  }
  let pct = 0;
  if (allHaveTotal && totalBytes > 0) {
    pct = (loadedBytes / totalBytes) * 100;
  } else {
    // Fall back to averaging per-file progress
    let sum = 0;
    for (const row of files.values()) sum += row.done ? 100 : row.progress;
    pct = sum / files.size;
  }
  pct = Math.max(0, Math.min(100, pct));
  downloadFill.style.width = `${pct}%`;
  downloadTotal.textContent = `${pct.toFixed(0)}%`;
}

function onProgress(raw: unknown) {
  const event = raw as ProgressEvent;
  if (event.status === 'ready') return;
  if (!('file' in event) || !event.file) return;

  const row = ensureFile(event.file);

  switch (event.status) {
    case 'initiate':
      row.meta.textContent = 'queued';
      break;
    case 'download':
      row.meta.textContent = 'downloading…';
      break;
    case 'progress': {
      const pct = event.progress ?? 0;
      row.progress = pct;
      row.loaded = event.loaded ?? 0;
      row.total = event.total ?? row.total;
      row.fill.style.width = `${Math.min(100, pct)}%`;
      if (row.total) {
        row.meta.textContent = `${formatBytes(row.loaded)} / ${formatBytes(row.total)}`;
      } else {
        row.meta.textContent = `${pct.toFixed(0)}%`;
      }
      break;
    }
    case 'done':
      row.done = true;
      row.progress = 100;
      row.li.dataset.done = 'true';
      if (row.total) {
        row.meta.textContent = formatBytes(row.total);
      } else {
        row.meta.textContent = 'done';
      }
      break;
  }
  updateTotals();
}

async function isModelCached(modelId: string): Promise<boolean> {
  if (!('caches' in globalThis)) return false;
  try {
    const cacheNames = ['transformers-cache', 'onnx-cache'];
    for (const name of cacheNames) {
      if (!(await caches.has(name))) continue;
      const cache = await caches.open(name);
      const requests = await cache.keys();
      if (requests.some((r) => r.url.includes(modelId))) return true;
    }
  } catch {
    /* ignore */
  }
  return false;
}

async function loadModel(fromCache: boolean) {
  setMode('busy', fromCache ? 'Warming up…' : 'Loading…');
  setStatus('loading', fromCache ? 'Loading from cache…' : 'Downloading weights…');

  // Only show the download card if the load actually takes time. If everything
  // is already cached, pipeline() resolves almost instantly and the card never
  // appears.
  downloadCard.hidden = true;
  const revealTimer = fromCache
    ? null
    : window.setTimeout(() => {
        downloadCard.hidden = false;
      }, 400);

  try {
    classifier = (await pipeline('token-classification', MODEL_ID, {
      device: 'webgpu',
      dtype: 'q4',
      progress_callback: fromCache ? undefined : onProgress,
    })) as unknown as ClassifierFn;

    if (revealTimer !== null) clearTimeout(revealTimer);

    if (!downloadCard.hidden) {
      downloadFill.style.width = '100%';
      downloadTotal.textContent = '100%';
      setTimeout(() => {
        downloadCard.hidden = true;
      }, 600);
    }

    setStatus('ready', 'Model ready · running on WebGPU');
    setMode('redact', 'Redact');
  } catch (err) {
    if (revealTimer !== null) clearTimeout(revealTimer);
    console.error(err);
    setStatus(
      'error',
      fromCache ? 'Cache load failed — click to re-download' : 'Model load failed — see console',
    );
    setMode('load', fromCache ? 'Load model' : 'Retry load');
    downloadCard.hidden = true;
  }
}

(async () => {
  if (await isModelCached(MODEL_ID)) {
    setStatus('loading', 'Model cached — initializing…');
    await loadModel(true);
  }
})();

function makeTag(ent: Entity): HTMLSpanElement {
  const tag = document.createElement('span');
  tag.className = 'redacted';
  tag.textContent = 'REDACTED';
  tag.title = `${ent.entity_group} · ${ent.word} · ${(ent.score * 100).toFixed(1)}%`;
  return tag;
}

function resolveSpan(input: string, ent: Entity, from: number): [number, number] | null {
  const sOk = typeof ent.start === 'number' && typeof ent.end === 'number'
    && ent.end > ent.start && ent.end <= input.length
    && input.slice(ent.start, ent.end).trim().length > 0;
  if (sOk) return [ent.start, ent.end];

  const word = (ent.word ?? '').replace(/^##/, '').trim();
  if (!word) return null;
  const idx = input.indexOf(word, from);
  if (idx < 0) return null;
  return [idx, idx + word.length];
}

function renderRedaction(input: string, entities: Entity[]) {
  outputArea.replaceChildren();
  if (!entities.length) {
    outputArea.append(document.createTextNode(input));
    return;
  }

  const spans: {start: number; end: number; ent: Entity}[] = [];
  let searchFrom = 0;
  for (const ent of entities) {
    const span = resolveSpan(input, ent, searchFrom);
    if (!span) continue;
    spans.push({start: span[0], end: span[1], ent});
    searchFrom = span[1];
  }
  spans.sort((a, b) => a.start - b.start);

  let cursor = 0;
  for (const s of spans) {
    if (s.start < cursor) continue;
    if (s.start > cursor) {
      outputArea.append(document.createTextNode(input.slice(cursor, s.start)));
    }
    outputArea.append(makeTag(s.ent));
    cursor = s.end;
  }
  if (cursor < input.length) {
    outputArea.append(document.createTextNode(input.slice(cursor)));
  }
}

async function runRedaction() {
  if (!classifier) return;
  const content = textarea.value.trim();
  if (!content) {
    textarea.focus();
    return;
  }

  setStatus('busy', 'Scanning tokens…');
  setMode('busy', 'Scanning…');
  stats.textContent = '';

  const t0 = performance.now();
  try {
    const entities = await classifier(content, {aggregation_strategy: 'simple'});
    const elapsed = performance.now() - t0;
    renderRedaction(content, entities);
    stats.textContent = `${entities.length} redacted · ${elapsed.toFixed(0)} ms`;
    setStatus('ready', 'Model ready · running on WebGPU');
  } catch (err) {
    console.error(err);
    setStatus('error', 'Inference failed — see console');
  } finally {
    setMode('redact', 'Redact');
  }
}

actionButton.addEventListener('click', () => {
  const mode = actionButton.dataset.mode;
  if (mode === 'busy') return;
  if (classifier) void runRedaction();
  else void loadModel(false);
});

textarea.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (classifier) void runRedaction();
  }
});

exampleButton.addEventListener('click', () => {
  textarea.value = EXAMPLE;
  textarea.focus();
});
