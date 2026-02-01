import { useState, useEffect, useCallback, useRef } from 'react'
import {
  getDefaultDataset,
  getFeatureRanges,
  uploadCsv,
  requestDecision,
  exportReport,
  submitFeedback,
  createStudySession,
  endStudySession,
  logInteraction,
  getStudyMetrics,
  getCurrentTask,
  submitTaskResponse,
  submitPostQuestionnaire,
  getDatasetCatalog,
  loadDataset,
  getRecentUploads,
  loadRecentUpload,
  requestDomainDecision,
} from './api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import './App.css'

const FEATURE_COLS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
const NUMERIC_DECISION_FACTORS = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']

// Study conditions
const STUDY_CONDITIONS = [
  { id: 'interactive', label: 'Interactive', desc: 'Full PRISM with all features' },
  { id: 'static', label: 'Static', desc: 'Basic explanations without interactivity' },
  { id: 'minimal', label: 'Minimal', desc: 'Decision only, no explanations (baseline)' },
]

// Study phases
const PHASES = {
  ONBOARDING: 'onboarding',
  PRE_QUESTIONNAIRE: 'pre_questionnaire',
  TASKS: 'tasks',
  EXPLORATION: 'exploration',
  POST_QUESTIONNAIRE: 'post_questionnaire',
  COMPLETE: 'complete',
}

export default function App() {
  // ============== DATA STATE ==============
  const [rows, setRows] = useState([])
  const [columns, setColumns] = useState(FEATURE_COLS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [result, setResult] = useState(null)
  const [deciding, setDeciding] = useState(false)
  const [featureRanges, setFeatureRanges] = useState({})
  const [whatIfRow, setWhatIfRow] = useState(null)
  
  // ============== DATASET PICKER STATE ==============
  const [datasetCatalog, setDatasetCatalog] = useState([])
  const [recentUploads, setRecentUploads] = useState([])
  const [showDatasetPicker, setShowDatasetPicker] = useState(false)
  const [activeDataset, setActiveDataset] = useState(null)

  // ============== STUDY SESSION STATE ==============
  const [studySession, setStudySession] = useState(null)
  const [studyPhase, setStudyPhase] = useState(PHASES.ONBOARDING)
  const [studyCondition, setStudyCondition] = useState('interactive')
  const [participantId, setParticipantId] = useState('')
  const [studyMetrics, setStudyMetrics] = useState(null)
  const [skippedQuestionnaires, setSkippedQuestionnaires] = useState(false)

  // ============== PRE-QUESTIONNAIRE STATE ==============
  const [preQ, setPreQ] = useState({
    age_range: null,
    gender: null,
    education: null,
    finance_experience: 3,
    credit_familiarity: 3,
    ai_familiarity: 3,
    ai_trust_general: 3,
    explainable_ai_exposure: false,
    tech_comfort: 3,
  })

  // ============== TASK STATE ==============
  const [currentTask, setCurrentTask] = useState(null)
  const [taskResponse, setTaskResponse] = useState('')
  const [taskConfidence, setTaskConfidence] = useState(3)
  const [taskStartTime, setTaskStartTime] = useState(null)
  const [tasksCompleted, setTasksCompleted] = useState(0)
  const [totalTasks, setTotalTasks] = useState(0)

  // ============== POST-QUESTIONNAIRE STATE (Trimmed) ==============
  const [postQ, setPostQ] = useState({
    // NASA-TLX: 3 key items for cognitive load
    nasa_tlx: { mental_demand: 10, effort: 10, frustration: 10 },
    // Trust: 4 key items for XAI evaluation
    trust: { trustworthy: 4, understandable: 4, reliable: 4, confident: 4 },
    // Usability: 2 key items
    usability: { easy_to_use: 3, would_use_again: 3 },
    // Open feedback: 2 items
    most_helpful_feature: '',
    improvement_suggestions: '',
  })

  // ============== UI STATE ==============
  const [explanationMode, setExplanationMode] = useState('plain')
  const [preferredMode, setPreferredMode] = useState(null)

  // ============== TRACKING STATE ==============
  const sessionStartTime = useRef(null)
  const lastInteractionTime = useRef(null)
  const sectionTimes = useRef({})
  const currentSection = useRef(null)
  const hoverStartTime = useRef(null)
  const scrollDepths = useRef([])

  // ============== INTERACTION TRACKING ==============
  const trackInteraction = useCallback(async (action, details = {}) => {
    if (!studySession?.session_id) return
    lastInteractionTime.current = Date.now()
    try {
      await logInteraction(studySession.session_id, action, {
        ...details,
        time_since_session_start_ms: sessionStartTime.current ? Date.now() - sessionStartTime.current : 0,
        section_times: { ...sectionTimes.current },
      })
    } catch (e) {
      console.warn('Failed to log interaction:', e)
    }
  }, [studySession])

  // Track section time
  const trackSectionEnter = (section) => {
    if (currentSection.current && currentSection.current !== section) {
      const elapsed = Date.now() - (sectionTimes.current[`${currentSection.current}_start`] || Date.now())
      sectionTimes.current[currentSection.current] = (sectionTimes.current[currentSection.current] || 0) + elapsed
    }
    currentSection.current = section
    sectionTimes.current[`${section}_start`] = Date.now()
  }

  // Track hover time
  const handleHoverStart = () => {
    hoverStartTime.current = Date.now()
  }

  const handleHoverEnd = async (elementType) => {
    if (hoverStartTime.current) {
      const hoverTime = Date.now() - hoverStartTime.current
      await trackInteraction('hover', { element: elementType, hover_time_ms: hoverTime })
      hoverStartTime.current = null
    }
  }

  // Track scroll depth
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      const scrollPercent = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0
      scrollDepths.current.push(scrollPercent)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // ============== SESSION MANAGEMENT ==============
  const startPreQuestionnaire = () => {
    if (!participantId.trim()) {
      setError('Please enter a participant ID')
      return
    }
    setStudyPhase(PHASES.PRE_QUESTIONNAIRE)
  }

  const skipQuestionnaires = async () => {
    if (!participantId.trim()) {
      setError('Please enter a participant ID')
      return
    }
    setError('')
    try {
      const session = await createStudySession(participantId.trim(), studyCondition, null)
      setStudySession(session)
      setSkippedQuestionnaires(true)
      sessionStartTime.current = Date.now()
      setExplanationMode(studyCondition === 'static' ? 'static' : studyCondition === 'minimal' ? 'minimal' : 'plain')
      setStudyPhase(PHASES.EXPLORATION)

      const dataResult = await getDefaultDataset(80)
      setColumns(dataResult.columns || FEATURE_COLS)
      setRows(dataResult.rows || [])
      const ranges = dataResult.decision_factor_ranges ?? dataResult.feature_ranges ?? {}
      if (Object.keys(ranges).length) setFeatureRanges(ranges)
      await loadRanges()
    } catch (e) {
      setError(e.message)
    }
  }

  const submitPreQuestionnaire = async () => {
    try {
      const session = await createStudySession(participantId.trim(), studyCondition, preQ)
      setStudySession(session)
      sessionStartTime.current = Date.now()
      setExplanationMode(studyCondition === 'static' ? 'static' : studyCondition === 'minimal' ? 'minimal' : 'plain')
      setStudyPhase(PHASES.TASKS)
      
      // Load data and get the rows directly
      const dataResult = await getDefaultDataset(80)
      const loadedRows = dataResult.rows || []
      setColumns(dataResult.columns || FEATURE_COLS)
      setRows(loadedRows)
      const ranges = dataResult.decision_factor_ranges ?? dataResult.feature_ranges ?? {}
      if (Object.keys(ranges).length) setFeatureRanges(ranges)
      
      await loadRanges()
      
      // Get first task
      const task = await getCurrentTask(session.session_id)
      if (task && !task.completed) {
        setCurrentTask(task)
        setTotalTasks(task.total_tasks || 4)
        setTaskStartTime(Date.now())
        // Auto-fetch decision for the first task's row
        if (task.row_index >= 0 && loadedRows[task.row_index]) {
          await runDecision(loadedRows[task.row_index])
        }
      } else {
        setStudyPhase(PHASES.EXPLORATION)
      }
    } catch (e) {
      setError(e.message)
    }
  }

  const handleTaskSubmit = async () => {
    if (!taskResponse.trim()) {
      setError('Please provide a response')
      return
    }
    
    const timeTaken = (Date.now() - taskStartTime) / 1000
    try {
      const taskResult = await submitTaskResponse(
        studySession.session_id,
        currentTask.task_id,
        taskResponse,
        taskConfidence,
        timeTaken
      )
      
      setTasksCompleted(tc => tc + 1)
      setTaskResponse('')
      setTaskConfidence(3)
      setResult(null) // Clear previous decision result
      
      if (taskResult.tasks_remaining > 0) {
        const nextTask = await getCurrentTask(studySession.session_id)
        if (nextTask && !nextTask.completed) {
          setCurrentTask(nextTask)
          setTaskStartTime(Date.now())
          // Auto-fetch decision for the new task's row
          if (nextTask.row_index >= 0 && rows[nextTask.row_index]) {
            await runDecision(rows[nextTask.row_index])
          }
        } else {
          setStudyPhase(PHASES.EXPLORATION)
        }
      } else {
        setStudyPhase(PHASES.EXPLORATION)
      }
    } catch (e) {
      setError(e.message)
    }
  }

  const startPostQuestionnaire = () => {
    if (skippedQuestionnaires) {
      endStudyWithoutQuestionnaire()
    } else {
      setStudyPhase(PHASES.POST_QUESTIONNAIRE)
    }
  }

  const endStudyWithoutQuestionnaire = async () => {
    try {
      await trackInteraction('study_complete', { skipped_questionnaires: true })
      await endStudySession(studySession.session_id)
      const metrics = await getStudyMetrics(studySession.session_id)
      setStudyMetrics(metrics)
      setStudyPhase(PHASES.COMPLETE)
    } catch (e) {
      setError(e.message)
    }
  }

  const submitPostQuestionnaireAndEnd = async () => {
    try {
      await submitPostQuestionnaire(studySession.session_id, postQ)
      await trackInteraction('study_complete', {
        max_scroll_depth: Math.max(...scrollDepths.current, 0),
        preferred_mode: preferredMode,
      })
      await endStudySession(studySession.session_id)
      const metrics = await getStudyMetrics(studySession.session_id)
      setStudyMetrics(metrics)
      setStudyPhase(PHASES.COMPLETE)
    } catch (e) {
      setError(e.message)
    }
  }

  // ============== DATA LOADING ==============
  const loadDefault = useCallback(async () => {
    setError('')
    setLoading(true)
    setShowDatasetPicker(false)
    try {
      const d = await getDefaultDataset(80)
      setColumns(d.columns || FEATURE_COLS)
      setRows(d.rows || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      setActiveDataset({
        type: 'catalog',
        id: 'uci_credit_approval',
        name: 'UCI Credit Approval',
        modelCompatible: true,
      })
      const ranges = d.decision_factor_ranges ?? d.feature_ranges ?? {}
      if (Object.keys(ranges).length) setFeatureRanges(ranges)
      trackInteraction('load_dataset', { source: 'default', row_count: d.rows?.length || 0 })
    } catch (e) {
      setError(e.message)
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [trackInteraction])

  const loadRanges = useCallback(async () => {
    try {
      const r = await getFeatureRanges()
      if (r && Object.keys(r).length) setFeatureRanges(r)
    } catch (_) {}
  }, [])

  // Load dataset catalog and recent uploads
  const loadDatasetCatalog = useCallback(async () => {
    try {
      const [catalog, uploads] = await Promise.all([
        getDatasetCatalog(false),
        getRecentUploads(),
      ])
      setDatasetCatalog(catalog.datasets || [])
      setRecentUploads(uploads.uploads || [])
    } catch (e) {
      console.error('Failed to load dataset catalog:', e)
    }
  }, [])

  // Load a dataset from catalog
  const selectCatalogDataset = async (datasetId) => {
    setError('')
    setLoading(true)
    setShowDatasetPicker(false)
    try {
      const result = await loadDataset(datasetId, 80)
      setColumns(result.columns || FEATURE_COLS)
      setRows(result.rows || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      setActiveDataset({
        type: 'catalog',
        id: datasetId,
        name: result.info?.name || datasetId,
        modelCompatible: result.model_compatible,
      })
      const ranges = result.feature_ranges ?? {}
      if (Object.keys(ranges).length) setFeatureRanges(ranges)
      trackInteraction('load_dataset', { source: 'catalog', dataset_id: datasetId, row_count: result.rows?.length || 0 })
    } catch (e) {
      setError(e.message)
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  // Load a recent upload
  const selectRecentUpload = async (uploadId) => {
    setError('')
    setLoading(true)
    setShowDatasetPicker(false)
    try {
      const result = await loadRecentUpload(uploadId)
      setColumns(result.columns || FEATURE_COLS)
      setRows(result.rows || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      setActiveDataset({
        type: 'recent',
        id: uploadId,
        name: result.filename,
        modelCompatible: true, // User uploads are assumed compatible
      })
      loadRanges()
      trackInteraction('load_dataset', { source: 'recent_upload', upload_id: uploadId, row_count: result.row_count })
    } catch (e) {
      setError(e.message)
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  const onUpload = async (e) => {
    const f = e.target?.files?.[0]
    if (!f) return
    setError('')
    setLoading(true)
    setShowDatasetPicker(false)
    try {
      const u = await uploadCsv(f)
      if (!u.ok && u.errors?.length) throw new Error(u.errors.join('; '))
      setColumns(u.columns || FEATURE_COLS)
      setRows(u.sample || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      setActiveDataset({
        type: 'upload',
        id: u.upload_id,
        name: f.name,
        modelCompatible: true,
      })
      loadRanges()
      // Refresh recent uploads list
      getRecentUploads().then(r => setRecentUploads(r.uploads || [])).catch(() => {})
      trackInteraction('upload_dataset', { filename: f.name, row_count: u.row_count })
    } catch (e) {
      setError(e.message)
      setRows([])
    } finally {
      setLoading(false)
    }
    e.target.value = ''
  }

  const runDecision = useCallback(async (row, isWhatIf = false) => {
    setDeciding(true)
    setError('')
    const startTime = Date.now()
    try {
      // Use domain-aware decision if we have an active dataset with a domain ID
      let r
      if (activeDataset?.id && activeDataset.type === 'catalog') {
        r = await requestDomainDecision(activeDataset.id, row)
      } else {
        r = await requestDecision(row)
      }
      setResult(r)
      const elapsed = Date.now() - startTime
      trackInteraction('view_decision', {
        decision: r.decision?.decision,
        confidence: r.decision?.confidence,
        is_whatif: isWhatIf,
        response_time_ms: elapsed,
        domain_id: activeDataset?.id,
      })
      return r
    } catch (e) {
      setError(e.message)
    } finally {
      setDeciding(false)
    }
  }, [trackInteraction, activeDataset])

  const onSelectRow = async (row, index) => {
    setSelectedIndex(index)
    setResult(null)
    setWhatIfRow(null)
    trackInteraction('select_row', { row_index: index })
    trackSectionEnter('decision')
    await runDecision(row)
  }

  const onWhatIfChange = (feature, value) => {
    if (studyCondition !== 'interactive') return
    const base = whatIfRow ?? (selectedIndex != null ? rows[selectedIndex] : null)
    if (!base) return
    const next = { ...base }
    next[feature] = value
    setWhatIfRow(next)
    trackInteraction('whatif_adjust', { feature, value, previous: base[feature] })
  }

  const onApplyWhatIf = async () => {
    const row = whatIfRow ?? (selectedIndex != null ? rows[selectedIndex] : null)
    if (!row) return
    setResult(null)
    trackInteraction('whatif_apply', { modified_features: Object.keys(whatIfRow || {}) })
    await runDecision(row, true)
  }

  const handleModeChange = (mode) => {
    if (studyCondition !== 'interactive') return
    const from = explanationMode
    setExplanationMode(mode)
    trackInteraction('change_mode', { from, to: mode })
    trackSectionEnter(mode)
  }

  const handleSetPreferredMode = (mode) => {
    setPreferredMode(mode)
    trackInteraction('set_preferred_mode', { mode })
  }

  const handleExport = async (format) => {
    if (!result) return
    try {
      await exportReport(format, result)
      trackInteraction('export', { format })
    } catch (e) {
      setError(e.message)
    }
  }

  // Prepare SHAP visualization data
  const names = result?.shap?.decision_factor_names ?? []
  const shapData = names.length
    ? names.map((n, i) => ({
        name: String(n).length > 24 ? String(n).slice(-20) + '…' : n,
        value: result.shap.values[i],
        full: n,
      }))
    : []

  const activeRow = whatIfRow ?? (selectedIndex != null ? rows[selectedIndex] : null)
  const isStaticMode = studyCondition === 'static'
  const isMinimalMode = studyCondition === 'minimal'
  const trustCal = result?.trust_calibration

  // ============== RENDER: ONBOARDING ==============
  if (studyPhase === PHASES.ONBOARDING) {
    return (
      <div className="app onboarding">
        <header className="header">
          <h1>PRISM</h1>
          <p className="tagline">Human-centred Explainable AI for Credit Decisions</p>
        </header>

        <section className="section onboarding-card">
          <h2>Welcome to the PRISM User Study</h2>
          <p className="study-intro">
            This study evaluates how well PRISM helps you understand AI-driven credit decisions.
            You will complete tasks, review decisions, and provide feedback.
          </p>

          <div className="form-row">
            <label>Participant ID *</label>
            <input
              type="text"
              value={participantId}
              onChange={(e) => setParticipantId(e.target.value)}
              placeholder="Enter your assigned participant ID"
            />
          </div>

          <div className="form-row">
            <label>Study Condition</label>
            <div className="condition-selector">
              {STUDY_CONDITIONS.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={studyCondition === c.id ? 'active' : ''}
                  onClick={() => setStudyCondition(c.id)}
                >
                  {c.label}
                </button>
              ))}
            </div>
            <p className="muted condition-desc">
              {STUDY_CONDITIONS.find(c => c.id === studyCondition)?.desc}
            </p>
          </div>

          {error && <div className="banner error">{error}</div>}

          <div className="onboarding-actions">
            <button type="button" className="start-study-btn" onClick={startPreQuestionnaire}>
              Continue to Background Questions
            </button>
            <button type="button" className="skip-btn" onClick={skipQuestionnaires}>
              Skip Questionnaires — Go to App
            </button>
          </div>

          <p className="muted ethics-note">
            By proceeding, you consent to participate in this research study.
            Your interactions will be logged for analysis. No personal data is collected.
          </p>
        </section>
      </div>
    )
  }

  // ============== RENDER: PRE-QUESTIONNAIRE ==============
  if (studyPhase === PHASES.PRE_QUESTIONNAIRE) {
    return (
      <div className="app questionnaire">
        <header className="header">
          <h1>PRISM</h1>
          <p className="tagline">Background Questionnaire</p>
        </header>

        <section className="section questionnaire-card">
          <h2>About You</h2>
          <p className="muted">Please answer a few background questions. This helps us understand our participants.</p>

          <div className="q-grid">
            <div className="form-row">
              <label>Age Range</label>
              <select value={preQ.age_range || ''} onChange={(e) => setPreQ(p => ({ ...p, age_range: e.target.value || null }))}>
                <option value="">Select...</option>
                <option value="18-24">18-24</option>
                <option value="25-34">25-34</option>
                <option value="35-44">35-44</option>
                <option value="45-54">45-54</option>
                <option value="55-64">55-64</option>
                <option value="65+">65+</option>
              </select>
            </div>

            <div className="form-row">
              <label>Gender</label>
              <select value={preQ.gender || ''} onChange={(e) => setPreQ(p => ({ ...p, gender: e.target.value || null }))}>
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="non-binary">Non-binary</option>
                <option value="prefer_not_to_say">Prefer not to say</option>
              </select>
            </div>

            <div className="form-row">
              <label>Education Level</label>
              <select value={preQ.education || ''} onChange={(e) => setPreQ(p => ({ ...p, education: e.target.value || null }))}>
                <option value="">Select...</option>
                <option value="high_school">High School</option>
                <option value="bachelors">Bachelor's Degree</option>
                <option value="masters">Master's Degree</option>
                <option value="doctorate">Doctorate</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <h3>Experience & Familiarity</h3>
          <p className="muted">Rate from 1 (lowest) to 5 (highest)</p>

          {[
            { key: 'finance_experience', label: 'Experience with financial decisions' },
            { key: 'credit_familiarity', label: 'Familiarity with credit/loan applications' },
            { key: 'ai_familiarity', label: 'Familiarity with AI/machine learning' },
            { key: 'ai_trust_general', label: 'General trust in AI systems' },
            { key: 'tech_comfort', label: 'Comfort with technology' },
          ].map(({ key, label }) => (
            <div key={key} className="form-row slider-input">
              <label>{label}: <strong>{preQ[key]}</strong></label>
              <input
                type="range"
                min={1}
                max={5}
                value={preQ[key]}
                onChange={(e) => setPreQ(p => ({ ...p, [key]: parseInt(e.target.value) }))}
              />
              <div className="range-labels"><span>1 (Low)</span><span>5 (High)</span></div>
            </div>
          ))}

          <div className="form-row checkbox-row">
            <label>
              <input
                type="checkbox"
                checked={preQ.explainable_ai_exposure}
                onChange={(e) => setPreQ(p => ({ ...p, explainable_ai_exposure: e.target.checked }))}
              />
              I have used explainable AI systems before
            </label>
          </div>

          {error && <div className="banner error">{error}</div>}

          <button type="button" className="start-study-btn" onClick={submitPreQuestionnaire}>
            Start Study Tasks
          </button>
        </section>
      </div>
    )
  }

  // ============== RENDER: TASK PHASE ==============
  if (studyPhase === PHASES.TASKS && currentTask) {
    return (
      <div className="app task-phase">
        <header className="header">
          <h1>PRISM</h1>
          <p className="tagline">Study Task</p>
          <div className="progress-indicator">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${((tasksCompleted + 1) / totalTasks) * 100}%` }} />
            </div>
            <span className="progress-text">Task {tasksCompleted + 1} of {totalTasks}</span>
          </div>
        </header>

        <section className="section task-card">
          {currentTask.task_type === 'attention_check' ? (
            <div className="attention-check">
              <span className="task-badge attention">Attention Check</span>
              <h2>{currentTask.question}</h2>
            </div>
          ) : (
            <>
              <span className="task-badge">{currentTask.task_type}</span>
              <h2>{currentTask.question}</h2>
              {currentTask.row_index >= 0 && rows[currentTask.row_index] && (
                <div className="task-data-preview">
                  <p className="muted">Reviewing application #{currentTask.row_index + 1}</p>
                  <button type="button" onClick={() => onSelectRow(rows[currentTask.row_index], currentTask.row_index)}>
                    View Full Decision
                  </button>
                </div>
              )}
            </>
          )}

          {currentTask.options ? (
            <div className="task-options">
              {currentTask.options.map((opt, i) => (
                <button
                  key={i}
                  type="button"
                  className={taskResponse === opt ? 'selected' : ''}
                  onClick={() => setTaskResponse(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>
          ) : (
            <div className="form-row">
              <textarea
                rows={3}
                value={taskResponse}
                onChange={(e) => setTaskResponse(e.target.value)}
                placeholder="Enter your response..."
              />
            </div>
          )}

          <div className="form-row slider-input">
            <label>Confidence in your answer: <strong>{taskConfidence}</strong></label>
            <input
              type="range"
              min={1}
              max={5}
              value={taskConfidence}
              onChange={(e) => setTaskConfidence(parseInt(e.target.value))}
            />
            <div className="range-labels"><span>1 (Not confident)</span><span>5 (Very confident)</span></div>
          </div>

          {error && <div className="banner error">{error}</div>}

          <button type="button" className="submit-task-btn" onClick={handleTaskSubmit} disabled={!taskResponse.trim()}>
            Submit Response
          </button>
        </section>

        {/* Show decision result if they clicked to view */}
        {result && (
          <section className="section decision-preview">
            <h3>PRISM Decision</h3>
            <p className="outcome">
              <span className={result.decision?.decision === '+' ? 'approved' : 'rejected'}>
                {result.decision?.decision === '+' ? 'Approved' : 'Rejected'}
              </span>
              {' '}(confidence: {((result.decision?.confidence ?? 0) * 100).toFixed(1)}%)
            </p>
            {studyCondition !== 'minimal' && result.explanation_layer?.bullets?.slice(0, 3).map((b, i) => (
              <p key={i} className="muted">• {b}</p>
            ))}
          </section>
        )}
      </div>
    )
  }

  // ============== RENDER: POST-QUESTIONNAIRE (Trimmed) ==============
  if (studyPhase === PHASES.POST_QUESTIONNAIRE) {
    return (
      <div className="app questionnaire post-q">
        <header className="header">
          <h1>PRISM</h1>
          <p className="tagline">Quick Feedback</p>
        </header>

        <section className="section questionnaire-card">
          <h2>Cognitive Load</h2>
          <p className="muted">Rate from 1 (very low) to 21 (very high)</p>

          {[
            { key: 'mental_demand', label: 'How mentally demanding was understanding the explanations?' },
            { key: 'effort', label: 'How hard did you have to work to make decisions?' },
            { key: 'frustration', label: 'How frustrated or stressed did you feel?' },
          ].map(({ key, label }) => (
            <div key={key} className="form-row slider-input">
              <label>{label}: <strong>{postQ.nasa_tlx[key]}</strong></label>
              <input
                type="range"
                min={1}
                max={21}
                value={postQ.nasa_tlx[key]}
                onChange={(e) => setPostQ(p => ({ ...p, nasa_tlx: { ...p.nasa_tlx, [key]: parseInt(e.target.value) } }))}
              />
              <div className="range-labels"><span>1 (Low)</span><span>21 (High)</span></div>
            </div>
          ))}

          <h2>Trust & Understanding</h2>
          <p className="muted">Rate from 1 (strongly disagree) to 7 (strongly agree)</p>

          {[
            { key: 'trustworthy', label: 'I trust the decisions made by this system' },
            { key: 'understandable', label: 'The explanations helped me understand why decisions were made' },
            { key: 'reliable', label: 'The system seems reliable' },
            { key: 'confident', label: 'I felt confident interpreting the results' },
          ].map(({ key, label }) => (
            <div key={key} className="form-row slider-input">
              <label>{label}: <strong>{postQ.trust[key]}</strong></label>
              <input
                type="range"
                min={1}
                max={7}
                value={postQ.trust[key]}
                onChange={(e) => setPostQ(p => ({ ...p, trust: { ...p.trust, [key]: parseInt(e.target.value) } }))}
              />
              <div className="range-labels"><span>1 (Disagree)</span><span>7 (Agree)</span></div>
            </div>
          ))}

          <h2>Usability</h2>
          <p className="muted">Rate from 1 (strongly disagree) to 5 (strongly agree)</p>

          {[
            { key: 'easy_to_use', label: 'The system was easy to use' },
            { key: 'would_use_again', label: 'I would use this system again' },
          ].map(({ key, label }) => (
            <div key={key} className="form-row slider-input">
              <label>{label}: <strong>{postQ.usability[key]}</strong></label>
              <input
                type="range"
                min={1}
                max={5}
                value={postQ.usability[key]}
                onChange={(e) => setPostQ(p => ({ ...p, usability: { ...p.usability, [key]: parseInt(e.target.value) } }))}
              />
              <div className="range-labels"><span>1 (Disagree)</span><span>5 (Agree)</span></div>
            </div>
          ))}

          <h2>Open Feedback</h2>
          <div className="form-row">
            <label>What was most helpful? (optional)</label>
            <textarea
              rows={2}
              value={postQ.most_helpful_feature}
              onChange={(e) => setPostQ(p => ({ ...p, most_helpful_feature: e.target.value }))}
              placeholder="e.g., the plain language explanations, what-if scenarios..."
            />
          </div>
          <div className="form-row">
            <label>Any suggestions for improvement? (optional)</label>
            <textarea
              rows={2}
              value={postQ.improvement_suggestions}
              onChange={(e) => setPostQ(p => ({ ...p, improvement_suggestions: e.target.value }))}
              placeholder="What could make this system better?"
            />
          </div>

          {error && <div className="banner error">{error}</div>}

          <button type="button" className="start-study-btn" onClick={submitPostQuestionnaireAndEnd}>
            Submit & Complete Study
          </button>
        </section>
      </div>
    )
  }

  // ============== RENDER: STUDY COMPLETE ==============
  if (studyPhase === PHASES.COMPLETE && studyMetrics) {
    return (
      <div className="app study-complete">
        <header className="header">
          <h1>PRISM</h1>
          <p className="tagline">Study Complete - Thank You!</p>
        </header>

        <section className="section">
          <h2>Your Session Summary</h2>

          <div className="metrics-card">
            <div className="metric">
              <span className="metric-value">{studyMetrics.tasks_completed}</span>
              <span className="metric-label">Tasks Completed</span>
            </div>
            <div className="metric">
              <span className="metric-value">{studyMetrics.task_accuracy?.toFixed(0) || 'N/A'}%</span>
              <span className="metric-label">Task Accuracy</span>
            </div>
            <div className="metric">
              <span className="metric-value">{studyMetrics.decisions_viewed}</span>
              <span className="metric-label">Decisions Viewed</span>
            </div>
            <div className="metric">
              <span className="metric-value">{Math.round(studyMetrics.duration_seconds / 60)}m</span>
              <span className="metric-label">Duration</span>
            </div>
            <div className="metric">
              <span className="metric-value">{studyMetrics.whatif_explorations}</span>
              <span className="metric-label">What-if Explorations</span>
            </div>
            <div className="metric">
              <span className="metric-value">{studyMetrics.condition}</span>
              <span className="metric-label">Condition</span>
            </div>
          </div>

          <p className="muted center-text">
            Session ID: {studyMetrics.session_id}<br/>
            Participant: {studyMetrics.participant_id}
          </p>

          <p className="thank-you-message">
            Your responses have been recorded. Thank you for contributing to this research on human-centred AI explanations.
          </p>
        </section>
      </div>
    )
  }

  // ============== RENDER: EXPLORATION PHASE (MAIN APP) ==============
  return (
    <div className={`app ${isStaticMode ? 'static-mode' : ''} ${isMinimalMode ? 'minimal-mode' : ''}`}>
      <header className="header">
        <h1>PRISM</h1>
        <p className="tagline">
          {isMinimalMode
            ? 'PRISM — AI Credit Decision System'
            : isStaticMode
            ? 'PRISM — AI Credit Decision System with Explanations'
            : 'PRISM — Human-centred Explainable AI'}
        </p>
        {studySession && (
          <div className="session-indicator">
            <span className={`condition-badge ${studyCondition}`}>
              {studyCondition === 'interactive' ? '🔬 Interactive' : studyCondition === 'static' ? '📊 Static' : '📋 Minimal'}
            </span>
            <div className="progress-mini">
              Reviewed: {selectedIndex !== null ? selectedIndex + 1 : 0} / {rows.length}
            </div>
            <button type="button" className="end-study-btn" onClick={startPostQuestionnaire}>
              Complete Study
            </button>
          </div>
        )}
      </header>

      {error && (
        <div className="banner error" role="alert">
          {error}
          <button type="button" onClick={() => setError('')} aria-label="Dismiss">×</button>
        </div>
      )}

      <section className="section">
        <h2>Dataset</h2>
        
        {/* Active dataset info */}
        {activeDataset && (
          <div className="active-dataset-banner">
            <span className="dataset-name">{activeDataset.name}</span>
            {activeDataset.modelCompatible ? (
              <span className="compatibility-badge compatible">Model Compatible</span>
            ) : (
              <span className="compatibility-badge incompatible">Exploration Only</span>
            )}
            <button type="button" className="change-dataset-btn" onClick={() => { loadDatasetCatalog(); setShowDatasetPicker(true) }}>
              Change Dataset
            </button>
          </div>
        )}
        
        {/* Dataset picker actions */}
        <div className="dataset-actions">
          <button type="button" onClick={loadDefault} disabled={loading}>
            Load default (UCI Credit)
          </button>
          <button 
            type="button" 
            onClick={() => { loadDatasetCatalog(); setShowDatasetPicker(!showDatasetPicker) }} 
            disabled={loading}
            className={showDatasetPicker ? 'active' : ''}
          >
            Browse Datasets
          </button>
          <label className="button-like">
            Upload CSV
            <input type="file" accept=".csv" onChange={onUpload} disabled={loading} hidden />
          </label>
        </div>
        
        {/* Dataset picker dropdown */}
        {showDatasetPicker && (
          <div className="dataset-picker">
            {/* Pre-loaded datasets */}
            <div className="picker-section">
              <h4>Credit Datasets</h4>
              <div className="dataset-grid">
                {datasetCatalog.map(ds => (
                  <div 
                    key={ds.id} 
                    className={`dataset-card ${ds.model_compatible ? 'compatible' : 'exploration'}`}
                    onClick={() => selectCatalogDataset(ds.id)}
                  >
                    <div className="dataset-card-header">
                      <span className="dataset-title">{ds.name}</span>
                      {ds.model_compatible ? (
                        <span className="badge-small compatible">PRISM Ready</span>
                      ) : (
                        <span className="badge-small exploration">Explore</span>
                      )}
                    </div>
                    <p className="dataset-desc">{ds.description}</p>
                    <div className="dataset-meta">
                      <span>{ds.rows?.toLocaleString()} rows</span>
                      <span>{ds.features} features</span>
                      <span className="source">{ds.source}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Recent uploads */}
            {recentUploads.length > 0 && (
              <div className="picker-section">
                <h4>Recent Uploads</h4>
                <div className="recent-uploads-list">
                  {recentUploads.map(upload => (
                    <div 
                      key={upload.id} 
                      className="recent-upload-item"
                      onClick={() => selectRecentUpload(upload.id)}
                    >
                      <span className="upload-filename">{upload.filename}</span>
                      <span className="upload-meta">
                        {upload.row_count} rows • {upload.column_count} cols
                      </span>
                      <span className="upload-time">
                        {new Date(upload.uploaded_at).toLocaleDateString()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <button 
              type="button" 
              className="close-picker-btn" 
              onClick={() => setShowDatasetPicker(false)}
            >
              Close
            </button>
          </div>
        )}
        
        {loading && <p className="muted">Loading…</p>}
        
        {/* Model compatibility warning */}
        {activeDataset && !activeDataset.modelCompatible && rows.length > 0 && (
          <div className="banner warning">
            <strong>Exploration Mode:</strong> This dataset has a different schema than the trained model. 
            You can explore the data, but PRISM decisions may not be accurate.
          </div>
        )}
        
        {rows.length > 0 ? (
          <div className="table-wrap" onMouseEnter={() => trackSectionEnter('dataset')}>
            <p className="muted">Select a row to review the PRISM decision. Showing {Math.min(rows.length, 30)} of {rows.length} rows.</p>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {columns.slice(0, 6).map((c) => <th key={c}>{c}</th>)}
                  <th>…</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 30).map((row, i) => (
                  <tr
                    key={i}
                    className={selectedIndex === i ? 'selected' : ''}
                    onClick={() => onSelectRow(row, i)}
                  >
                    <td>{i + 1}</td>
                    {columns.slice(0, 6).map((c) => <td key={c}>{String(row[c] ?? '')}</td>)}
                    <td>…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">Load a dataset to get started. Choose from pre-loaded credit datasets or upload your own CSV.</p>
        )}
      </section>

      {(selectedIndex != null || result) && (
      <section className="section" onMouseEnter={() => trackSectionEnter('explanation')}>
        <h2>Explanation</h2>
        
        {/* Mode toggle - interactive only */}
        {studyCondition === 'interactive' && (
          <div className="explanation-mode-toggle">
            <span className="muted">Mode: </span>
            {['plain', 'technical', 'whatif'].map((m) => (
              <button
                key={m}
                type="button"
                className={explanationMode === m ? 'active' : ''}
                onClick={() => handleModeChange(m)}
                onMouseEnter={handleHoverStart}
                onMouseLeave={() => handleHoverEnd(`mode_${m}`)}
              >
                {m === 'plain' ? 'Plain Language' : m === 'technical' ? 'Technical' : 'What-If'}
              </button>
            ))}
            
            {/* Preferred mode selector */}
            <span className="pref-label muted">| Preferred:</span>
            {['plain', 'technical', 'whatif'].map((m) => (
              <button
                key={`pref_${m}`}
                type="button"
                className={`pref-btn ${preferredMode === m ? 'preferred' : ''}`}
                onClick={() => handleSetPreferredMode(m)}
                title={`Set ${m} as preferred`}
              >
                {preferredMode === m ? '★' : '☆'}
              </button>
            ))}
          </div>
        )}

        {deciding && <p className="muted">Computing decision…</p>}

        {result && !deciding && (
          <>
            {/* Trust Calibration - interactive only */}
            {studyCondition === 'interactive' && trustCal && (
              <div 
                className={`trust-calibration-card band-${result.uncertainty?.confidence_band || 'medium'}`}
                onMouseEnter={handleHoverStart}
                onMouseLeave={() => handleHoverEnd('trust_calibration')}
              >
                <div className="trust-header">
                  <span className="trust-label">Model Reliability</span>
                  <span className="trust-accuracy">
                    Historical accuracy: <strong>{((trustCal.historical_accuracy || 0) * 100).toFixed(0)}%</strong>
                  </span>
                </div>
                {trustCal.calibration_warning && (
                  <p className="calibration-warning">{trustCal.calibration_warning}</p>
                )}
                <div className="complexity-indicator">
                  <span className="muted">Complexity:</span>
                  <div className="complexity-bar">
                    <div className="complexity-fill" style={{ width: `${(trustCal.complexity_score || 0.5) * 100}%` }} />
                  </div>
                  <span className="muted">~{trustCal.estimated_read_time_seconds || 30}s read</span>
                </div>
              </div>
            )}

            {/* Decision Card */}
            <div className="decision-card" onMouseEnter={handleHoverStart} onMouseLeave={() => handleHoverEnd('decision')}>
              <h3>Decision</h3>
              {(() => {
                const dec = result.decision || {}
                const posLabel = dec.positive_label || 'Approved'
                const negLabel = dec.negative_label || 'Rejected'
                const decision = dec.decision || ''
                const isPositive = decision === '+' || decision === posLabel
                const displayDecision = decision === '+' ? 'Approved' : decision === '-' ? 'Rejected' : decision
                const probs = dec.probabilities || {}
                // Get probabilities - try domain labels first, then fallback to +/-
                const posProb = probs[posLabel] ?? probs['+'] ?? 0
                const negProb = probs[negLabel] ?? probs['-'] ?? 0
                
                return (
                  <>
                    <p className="outcome">
                      <span className={isPositive ? 'approved' : 'rejected'}>
                        {displayDecision}
                      </span>
                      {' '}(confidence: {((dec.confidence ?? 0) * 100).toFixed(1)}%)
                    </p>
                    {!isMinimalMode && (
                      <p className="muted">
                        P({posLabel}) = {(posProb * 100).toFixed(1)}%,
                        P({negLabel}) = {(negProb * 100).toFixed(1)}%
                      </p>
                    )}
                  </>
                )
              })()}
            </div>

            {/* MINIMAL MODE: Just decision */}
            {isMinimalMode && (
              <p className="muted minimal-note">This is the baseline condition showing only the decision outcome.</p>
            )}

            {/* STATIC MODE: Simple explanation */}
            {isStaticMode && result.explanation_layer && (
              <div className="static-explanation-card">
                <h3>Key Factors</h3>
                <ul className="static-factors">
                  {result.explanation_layer.bullets?.slice(0, 3).map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* INTERACTIVE MODE: Full features */}
            {studyCondition === 'interactive' && (
              <>
                {/* Uncertainty */}
                {result.uncertainty && (
                  <div 
                    className={`uncertainty-card band-${result.uncertainty.confidence_band}`}
                    onMouseEnter={handleHoverStart}
                    onMouseLeave={() => handleHoverEnd('uncertainty')}
                  >
                    <h3>Uncertainty & Stability</h3>
                    <p className="confidence-band">
                      Confidence: <strong>{result.uncertainty.confidence_band}</strong>
                      {result.uncertainty.stable ? ' · Stable' : ' · Unstable'}
                    </p>
                    {result.uncertainty.warning && (
                      <div className="uncertainty-warning">{result.uncertainty.warning}</div>
                    )}
                  </div>
                )}

                {/* Plain language */}
                {(explanationMode === 'plain' || explanationMode === 'whatif') && result.explanation_layer && (
                  <div 
                    className="explanation-layer-card"
                    onMouseEnter={handleHoverStart}
                    onMouseLeave={() => handleHoverEnd('explanation_layer')}
                  >
                    <h3>Explanation</h3>
                    <p className="directional-reasoning">{result.explanation_layer.directional_reasoning}</p>
                    <ul className="plain-language-bullets">
                      {result.explanation_layer.bullets?.map((b, i) => <li key={i}>{b}</li>)}
                    </ul>
                  </div>
                )}

                {/* Counterfactual preview */}
                {result.counterfactual_preview?.length > 0 && explanationMode !== 'technical' && (
                  <div 
                    className="counterfactual-preview-card"
                    onMouseEnter={handleHoverStart}
                    onMouseLeave={() => handleHoverEnd('counterfactual')}
                  >
                    <h3>What Could Change the Outcome?</h3>
                    <ul>
                      {result.counterfactual_preview.map((p, i) => <li key={i}>{p.suggestion}</li>)}
                    </ul>
                  </div>
                )}

                {/* What-if sliders */}
                {explanationMode === 'whatif' && Object.keys(featureRanges).length > 0 && activeRow && (
                  <div className="whatif-card">
                    <h3>What-If Scenarios</h3>
                    <p className="muted">Adjust values and see how the decision changes.</p>
                    <div className="sliders">
                      {NUMERIC_DECISION_FACTORS.filter((f) => featureRanges[f]).map((f) => {
                        const r = featureRanges[f]
                        const min = r.min ?? 0
                        const max = Math.max(r.max ?? 100, min + 1)
                        const raw = activeRow[f]
                        const val = typeof raw === 'number' && !Number.isNaN(raw) ? raw : (parseFloat(raw) || min)
                        const clamped = Math.min(max, Math.max(min, val))
                        const step = Number.isInteger(min) && Number.isInteger(max) ? 1 : (max - min) / 100
                        return (
                          <div key={f} className="slider-row">
                            <label>{r.label || f}: <strong>{typeof clamped === 'number' ? clamped.toFixed(1) : clamped}</strong></label>
                            <input
                              type="range"
                              min={min}
                              max={max}
                              step={step}
                              value={clamped}
                              onChange={(e) => onWhatIfChange(f, parseFloat(e.target.value))}
                            />
                          </div>
                        )
                      })}
                    </div>
                    <button type="button" className="whatif-apply" onClick={onApplyWhatIf} disabled={!whatIfRow}>
                      Update Decision
                    </button>
                  </div>
                )}

                {/* Technical SHAP */}
                {explanationMode === 'technical' && shapData.length > 0 && (
                  <div 
                    className="shap-card"
                    onMouseEnter={handleHoverStart}
                    onMouseLeave={() => handleHoverEnd('shap_chart')}
                  >
                    <h3>SHAP Feature Impact</h3>
                    <p className="muted">Green = pushes toward approval, Red = pushes toward rejection</p>
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={Math.max(300, shapData.length * 22)}>
                        <BarChart data={[...shapData].reverse()} layout="vertical" margin={{ left: 8, right: 24 }}>
                          <XAxis type="number" />
                          <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 10 }} />
                          <Tooltip formatter={(v) => [v?.toFixed(4), 'SHAP']} />
                          <ReferenceLine x={0} stroke="#666" strokeDasharray="3 3" />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {[...shapData].reverse().map((d, i) => (
                              <Cell key={i} fill={d.value >= 0 ? '#22c55e' : '#ef4444'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Export */}
            {!isMinimalMode && (
              <div className="export-actions">
                <button type="button" onClick={() => handleExport('csv')}>Export CSV</button>
                <button type="button" onClick={() => handleExport('pdf')}>Export PDF</button>
              </div>
            )}
          </>
        )}
      </section>
      )}

      <footer className="footer">
        <p>PRISM — Human-centred Explainable AI for Credit Decisions</p>
        {studySession && <p className="muted">Session: {studySession.session_id}</p>}
      </footer>
    </div>
  )
}
