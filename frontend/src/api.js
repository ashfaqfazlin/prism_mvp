const BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '')}/api`
  : '/api';

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const t = await res.text();
    let msg = t;
    try {
      const j = JSON.parse(t);
      msg = j.detail || t;
    } catch {}
    throw new Error(msg);
  }
  if (res.headers.get('content-type')?.includes('application/json')) return res.json();
  return res;
}

export async function health() {
  return req('/health');
}

export async function getDefaultDataset(limit = 50) {
  return req(`/datasets/default?limit=${limit}`);
}

// ============== DATASET CATALOG API ==============

/** Get list of available pre-loaded datasets */
export async function getDatasetCatalog(featuredOnly = false) {
  return req(`/datasets/catalog?featured_only=${featuredOnly}`);
}

/** Load a specific dataset by ID */
export async function loadDataset(datasetId, limit = 80) {
  return req(`/datasets/${datasetId}?limit=${limit}`);
}

/** Get dataset-level (global) explainability: mean absolute SHAP across sample */
export async function getGlobalExplainability(datasetId, sampleSize = 50) {
  return req(`/datasets/${datasetId}/global-explainability?sample_size=${sampleSize}`);
}

/** Get list of recent user uploads */
export async function getRecentUploads() {
  return req('/datasets/recent/uploads');
}

/** Load a recent upload by ID */
export async function loadRecentUpload(uploadId) {
  return req(`/datasets/recent/${uploadId}`);
}

/** Clear all recent uploads */
export async function clearRecentUploads() {
  return req('/datasets/recent', { method: 'DELETE' });
}

// ============== UPLOAD TRAINING API ==============

/** Analyze an uploaded dataset to detect schema and suggest target */
export async function analyzeUpload(uploadId) {
  return req(`/datasets/upload/${uploadId}/analyze`);
}

/** Get information about a potential target column */
export async function getTargetInfo(uploadId, targetCol) {
  return req(`/datasets/upload/${uploadId}/target-info?target_col=${encodeURIComponent(targetCol)}`);
}

/** Configure an upload for training */
export async function configureUpload(uploadId, config) {
  return req(`/datasets/upload/${uploadId}/configure`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/** Train a model on the configured upload */
export async function trainUpload(uploadId, asyncMode = false) {
  return req(`/datasets/upload/${uploadId}/train?async_mode=${asyncMode}`, {
    method: 'POST',
  });
}

/** Get training status for an upload */
export async function getTrainingStatus(uploadId) {
  return req(`/datasets/upload/${uploadId}/status`);
}

/** List all trained dynamic domains */
export async function listTrainedUploads() {
  return req('/datasets/upload/trained');
}

/** Delete a dynamic domain and its trained model */
export async function deleteUploadDomain(uploadId) {
  return req(`/datasets/upload/${uploadId}`, { method: 'DELETE' });
}

// ============== DOMAIN/MODEL API ==============

/** Get list of available domains with training status */
export async function listDomains() {
  return req('/domains');
}

/** Get info about a specific domain */
export async function getDomainInfo(domainId) {
  return req(`/domains/${domainId}`);
}

/** Activate a specific domain/model */
export async function activateDomain(domainId) {
  return req(`/domains/${domainId}/activate`, { method: 'POST' });
}

/** Request decision for a specific domain */
export async function requestDomainDecision(domainId, row) {
  return req(`/decision/${domainId}`, { method: 'POST', body: JSON.stringify(row) });
}

/** Batch predict outcomes for multiple rows (fast preview, no full explanations) */
export async function requestBatchPredictions(domainId, rows) {
  return req(`/decision/${domainId}/batch`, { method: 'POST', body: JSON.stringify(rows) });
}

export async function getFeatureRanges() {
  return req('/feature-ranges');
}

/** PRISM decision engine: request decision, confidence, decision factors. */
export async function requestDecision(row) {
  return req('/decision', { method: 'POST', body: JSON.stringify(row) });
}

export async function uploadCsv(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: fd });
  if (!res.ok) {
    const t = await res.text();
    let msg = t;
    try {
      const j = JSON.parse(t);
      msg = Array.isArray(j.detail) ? j.detail.join('; ') : (j.detail || t);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

/** @deprecated Use requestDecision. */
export async function predict(row) {
  return req('/predict', { method: 'POST', body: JSON.stringify(row) });
}

export async function exportReport(format, data) {
  const res = await fetch(`${BASE}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format, data }),
  });
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  const name = `prism_export.${format}`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Bulk export: rows + predictions as CSV */
export async function exportBulk({ columns, rows, predictions }) {
  const res = await fetch(`${BASE}/export/bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format: 'csv', columns, rows, predictions }),
  });
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'prism_bulk_export.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

export async function submitFeedback(payload) {
  return req('/feedback', { method: 'POST', body: JSON.stringify(payload) });
}

// ============== STUDY MANAGEMENT API ==============

/** Create a new study session with optional pre-questionnaire and within-subjects flag */
export async function createStudySession(participantId, condition, preQuestionnaire = null, withinSubjects = false) {
  return req('/study/session', {
    method: 'POST',
    body: JSON.stringify({
      participant_id: participantId,
      condition,
      pre_questionnaire: preQuestionnaire,
      within_subjects: withinSubjects,
    }),
  });
}

/** Get study session details */
export async function getStudySession(sessionId) {
  return req(`/study/session/${sessionId}`);
}

/** End a study session */
export async function endStudySession(sessionId) {
  return req(`/study/session/${sessionId}/end`, { method: 'POST' });
}

/** Log a user interaction with rich tracking data */
export async function logInteraction(sessionId, action, details = {}) {
  return req('/study/interaction', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      action,
      details,
      timestamp: new Date().toISOString(),
    }),
  });
}

/** Get session metrics */
export async function getStudyMetrics(sessionId) {
  return req(`/study/session/${sessionId}/metrics`);
}

/** List all study sessions */
export async function listStudySessions() {
  return req('/study/sessions');
}

/** Export all study data */
export async function exportStudyData() {
  return req('/study/export');
}

// ============== TASK-BASED EVALUATION API ==============

/** Get current task for a session */
export async function getCurrentTask(sessionId) {
  return req(`/study/session/${sessionId}/task`);
}

/** Submit a task response */
export async function submitTaskResponse(sessionId, taskId, response, confidence = 3, timeTakenSeconds = 0) {
  return req(`/study/session/${sessionId}/task`, {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      task_id: taskId,
      response,
      confidence,
      time_taken_seconds: timeTakenSeconds,
    }),
  });
}

/** Submit post-study questionnaire */
export async function submitPostQuestionnaire(sessionId, questionnaire) {
  return req(`/study/session/${sessionId}/post-questionnaire`, {
    method: 'POST',
    body: JSON.stringify(questionnaire),
  });
}

/** Download study sessions CSV */
export function downloadStudySessionsCSV() {
  window.open(`${BASE}/study/export/csv`, '_blank');
}

/** Download interactions CSV */
export function downloadInteractionsCSV() {
  window.open(`${BASE}/study/export/interactions`, '_blank');
}
