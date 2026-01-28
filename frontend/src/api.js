const BASE = '/api';

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

export async function submitFeedback(payload) {
  return req('/feedback', { method: 'POST', body: JSON.stringify(payload) });
}

// ============== STUDY MANAGEMENT API ==============

/** Create a new study session with optional pre-questionnaire */
export async function createStudySession(participantId, condition, preQuestionnaire = null) {
  return req('/study/session', {
    method: 'POST',
    body: JSON.stringify({
      participant_id: participantId,
      condition,
      pre_questionnaire: preQuestionnaire,
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
