import { useState, useEffect, useCallback, useRef } from 'react'
import {
  getFeatureRanges,
  uploadCsv,
  requestDecision,
  exportReport,
  exportBulk,
  submitFeedback,
  getDatasetCatalog,
  loadDataset,
  getGlobalExplainability,
  getRecentUploads,
  loadRecentUpload,
  requestDomainDecision,
  requestBatchPredictions,
  analyzeUpload,
  getTargetInfo,
  configureUpload,
  trainUpload,
  getTrainingStatus,
  listTrainedUploads,
} from './api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import './App.css'

const DEFAULT_FEATURE_COLS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
// Max number of What-If sliders to show (for usability)
const MAX_WHATIF_SLIDERS = 8

// Demo mode only (SLR methodology: no user study, no participants, no questionnaires)
const DEMO_MODE = true // Always full interactive; no study conditions

export default function App() {
  // ============== DATA STATE ==============
  const [rows, setRows] = useState([])
  const [columns, setColumns] = useState(DEFAULT_FEATURE_COLS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [result, setResult] = useState(null)
  const [deciding, setDeciding] = useState(false)
  const [featureRanges, setFeatureRanges] = useState({})
  const [whatIfRow, setWhatIfRow] = useState(null)
  const [baselineResult, setBaselineResult] = useState(null) // Original prediction before What-If changes
  const [showComparison, setShowComparison] = useState(true) // Toggle comparison view in What-If mode
  const [rowPredictions, setRowPredictions] = useState({}) // Batch predictions for all rows {index: {decision, confidence}}
  const [predictingBatch, setPredictingBatch] = useState(false) // Loading state for batch predictions
  const [predictionFilter, setPredictionFilter] = useState('all') // 'all', 'positive', 'negative'
  const [tableSearch, setTableSearch] = useState('') // Search/filter text in table
  const [datasetSummary, setDatasetSummary] = useState(null) // { class_balance, row_count, feature_count, positive_count?, negative_count? }
  const [globalExplainability, setGlobalExplainability] = useState(null) // { feature_names, mean_abs_shap, feature_labels }
  const [compareIndex, setCompareIndex] = useState(null) // Second row for compare mode (null = single view)
  const [resultB, setResultB] = useState(null)
  const [decidingB, setDecidingB] = useState(false)
  const [bookmarks, setBookmarks] = useState(() => {
    try { return JSON.parse(localStorage.getItem('prism_bookmarks') || '[]') } catch { return [] }
  })
  const [showSavedCases, setShowSavedCases] = useState(false)
  const [pendingBookmark, setPendingBookmark] = useState(null) // Apply after dataset loads
  const [tourStep, setTourStep] = useState(null) // null | 0 | 1 | 2 | 3 | 4 (0=start, 4=done)
  const [theme, setTheme] = useState(() => localStorage.getItem('prism_theme') || 'dark') // 'dark' | 'light'
  const explanationSectionRef = useRef(null)
  
  // ============== DATASET PICKER STATE ==============
  const [datasetCatalog, setDatasetCatalog] = useState([])
  const [recentUploads, setRecentUploads] = useState([])
  const [showDatasetPicker, setShowDatasetPicker] = useState(false)
  const [activeDataset, setActiveDataset] = useState(null)
  
  // ============== TRAINING WIZARD STATE ==============
  const [showTrainingWizard, setShowTrainingWizard] = useState(false)
  const [trainingUpload, setTrainingUpload] = useState(null)
  const [trainingAnalysis, setTrainingAnalysis] = useState(null)
  const [trainingConfig, setTrainingConfig] = useState({
    targetCol: null,
    positiveValue: null,
    negativeValue: null,
    positiveLabel: 'Positive',
    negativeLabel: 'Negative',
    name: '',
  })
  const [targetInfo, setTargetInfo] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [trainingError, setTrainingError] = useState('')

  // ============== UI STATE (demo mode: full interactive explanations) ==============
  const [explanationMode, setExplanationMode] = useState('plain')

  // No-op for any legacy tracking; no participant/session data collected (SLR demo only)
  const trackInteraction = useCallback(async () => {}, [])

  const trackSectionEnter = () => {}

  const handleHoverStart = () => {}

  const handleHoverEnd = async () => {}

  // ============== DATA LOADING ==============
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

  // Run batch predictions for all rows
  const runBatchPredictions = useCallback(async (domainId, rowsData) => {
    if (!domainId || !rowsData?.length) return
    
    setPredictingBatch(true)
    setRowPredictions({})
    
    try {
      const result = await requestBatchPredictions(domainId, rowsData)
      const predictions = {}
      for (const p of result.predictions || []) {
        predictions[p.index] = {
          decision: p.decision,
          confidence: p.confidence,
          positiveLabel: p.positive_label || result.positive_label,
          negativeLabel: p.negative_label || result.negative_label,
          error: p.error,
        }
      }
      setRowPredictions(predictions)
    } catch (e) {
      console.error('Batch prediction failed:', e)
      // Don't show error - batch prediction is optional enhancement
    } finally {
      setPredictingBatch(false)
    }
  }, [])

  // Load a dataset from catalog
  const selectCatalogDataset = async (datasetId) => {
    setError('')
    setLoading(true)
    setShowDatasetPicker(false)
    setRowPredictions({}) // Clear previous predictions
    setPredictionFilter('all')
    try {
      const result = await loadDataset(datasetId, 80)
      setColumns(result.columns || DEFAULT_FEATURE_COLS)
      setRows(result.rows || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      setBaselineResult(null)
      setActiveDataset({
        type: 'catalog',
        id: datasetId,
        name: result.info?.name || datasetId,
        description: result.info?.description,
        modelCompatible: result.model_compatible,
        domain: result.info?.domain,
        accuracy: result.info?.accuracy,
        positiveLabel: result.info?.positive_label,
        negativeLabel: result.info?.negative_label,
      })
      const ranges = result.feature_ranges ?? {}
      if (Object.keys(ranges).length) setFeatureRanges(ranges)
      setDatasetSummary(result.summary || null)
      setCompareIndex(null)
      setResultB(null)
      trackInteraction('load_dataset', { source: 'catalog', dataset_id: datasetId, domain: result.info?.domain, row_count: result.rows?.length || 0 })
      
      // Run batch predictions if model is compatible
      if (result.model_compatible && result.rows?.length) {
        runBatchPredictions(datasetId, result.rows)
      }
      // Fetch global explainability (dataset-level feature importance)
      if (result.model_compatible) {
        getGlobalExplainability(datasetId, 50).then(setGlobalExplainability).catch(() => setGlobalExplainability(null))
      } else {
        setGlobalExplainability(null)
      }
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
      setColumns(result.columns || DEFAULT_FEATURE_COLS)
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
      setColumns(u.columns || DEFAULT_FEATURE_COLS)
      setRows(u.sample || [])
      setSelectedIndex(null)
      setResult(null)
      setWhatIfRow(null)
      
      // Refresh recent uploads list
      getRecentUploads().then(r => setRecentUploads(r.uploads || [])).catch(() => {})
      trackInteraction('upload_dataset', { filename: f.name, row_count: u.row_count })
      
      // Open training wizard for the uploaded file
      openTrainingWizard(u.upload_id, f.name, u.sample || [])
    } catch (e) {
      setError(e.message)
      setRows([])
    } finally {
      setLoading(false)
    }
    e.target.value = ''
  }
  
  // ============== TRAINING WIZARD FUNCTIONS ==============
  
  const openTrainingWizard = async (uploadId, filename, sampleRows) => {
    setTrainingError('')
    setTrainingStatus(null)
    setTargetInfo(null)
    setTrainingConfig({
      targetCol: null,
      positiveValue: null,
      negativeValue: null,
      positiveLabel: 'Positive',
      negativeLabel: 'Negative',
      name: filename.replace('.csv', ''),
    })
    
    try {
      // Analyze the upload
      const analysis = await analyzeUpload(uploadId)
      setTrainingAnalysis(analysis)
      setTrainingUpload({ id: uploadId, filename, sample: sampleRows })
      setShowTrainingWizard(true)
      
      // Auto-select suggested target
      if (analysis.suggested_target) {
        setTrainingConfig(prev => ({ ...prev, targetCol: analysis.suggested_target }))
        // Fetch target info
        const info = await getTargetInfo(uploadId, analysis.suggested_target)
        setTargetInfo(info)
        if (info.suggested_positive !== undefined) {
          setTrainingConfig(prev => ({
            ...prev,
            positiveValue: info.suggested_positive,
            negativeValue: info.suggested_negative,
            positiveLabel: info.suggested_labels?.positive_label || 'Positive',
            negativeLabel: info.suggested_labels?.negative_label || 'Negative',
          }))
        }
      }
    } catch (e) {
      setTrainingError(e.message)
    }
  }
  
  const handleTargetColChange = async (col) => {
    setTrainingConfig(prev => ({ ...prev, targetCol: col }))
    if (!col || !trainingUpload) return
    
    try {
      const info = await getTargetInfo(trainingUpload.id, col)
      setTargetInfo(info)
      if (info.suggested_positive !== undefined) {
        setTrainingConfig(prev => ({
          ...prev,
          positiveValue: info.suggested_positive,
          negativeValue: info.suggested_negative,
        }))
      }
    } catch (e) {
      setTrainingError(e.message)
    }
  }
  
  const startTraining = async () => {
    if (!trainingUpload || !trainingConfig.targetCol) {
      setTrainingError('Please select a target column')
      return
    }
    if (trainingConfig.positiveValue === null || trainingConfig.negativeValue === null) {
      setTrainingError('Please select positive and negative values')
      return
    }
    
    setTrainingError('')
    setTrainingStatus({ status: 'configuring', progress: 0 })
    
    try {
      // Configure the upload
      await configureUpload(trainingUpload.id, {
        target_col: trainingConfig.targetCol,
        positive_value: trainingConfig.positiveValue,
        negative_value: trainingConfig.negativeValue,
        positive_label: trainingConfig.positiveLabel,
        negative_label: trainingConfig.negativeLabel,
        name: trainingConfig.name,
      })
      
      setTrainingStatus({ status: 'training', progress: 10 })
      
      // Start training (synchronous for simplicity)
      const result = await trainUpload(trainingUpload.id, false)
      
      setTrainingStatus({
        status: result.status,
        progress: result.progress,
        accuracy: result.accuracy,
        error: result.error,
      })
      
      if (result.status === 'completed') {
        // Set as active dataset with model compatibility
        setActiveDataset({
          type: 'trained_upload',
          id: trainingUpload.id,
          domain: result.domain_id,
          name: trainingConfig.name,
          modelCompatible: true,
          accuracy: result.accuracy,
          positiveLabel: trainingConfig.positiveLabel,
          negativeLabel: trainingConfig.negativeLabel,
        })
        
        // Close wizard after short delay
        setTimeout(() => {
          setShowTrainingWizard(false)
        }, 2000)
      }
    } catch (e) {
      setTrainingError(e.message)
      setTrainingStatus({ status: 'failed', error: e.message })
    }
  }
  
  const skipTraining = () => {
    // Use dataset in exploration mode only
    setActiveDataset({
      type: 'upload',
      id: trainingUpload.id,
      name: trainingUpload.filename,
      modelCompatible: false,
    })
    setShowTrainingWizard(false)
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
        explanation_fidelity: r.explanation_fidelity,
      })
      return r
    } catch (e) {
      setError(e.message)
    } finally {
      setDeciding(false)
    }
  }, [trackInteraction, activeDataset])

  const runDecisionB = useCallback(async (row) => {
    setDecidingB(true)
    setError('')
    try {
      let r
      if (activeDataset?.id && activeDataset.type === 'catalog') {
        r = await requestDomainDecision(activeDataset.id, row)
      } else {
        r = await requestDecision(row)
      }
      setResultB(r)
      return r
    } catch (e) {
      setError(e.message)
    } finally {
      setDecidingB(false)
    }
  }, [activeDataset])

  const onSelectRow = async (row, index, isCompareClick = false) => {
    if (isCompareClick) {
      setCompareIndex(index)
      setResultB(null)
      const r = await runDecisionB(row)
      if (r) setResultB(r)
      return
    }
    setSelectedIndex(index)
    setResult(null)
    setWhatIfRow(null)
    setBaselineResult(null)
    setCompareIndex(null)
    setResultB(null)
    trackInteraction('select_row', { row_index: index })
    trackSectionEnter('decision')
    const r = await runDecision(row)
    if (r) setBaselineResult(r)
  }

  const onWhatIfChange = (feature, value) => {
    if (!DEMO_MODE) return
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
    // Run decision but preserve baseline for comparison
    await runDecision(row, true)
  }
  
  const onResetToBaseline = () => {
    setWhatIfRow(null)
    if (baselineResult) setResult(baselineResult)
    trackInteraction('whatif_reset', {})
  }

  // Apply a counterfactual suggestion to What-If sliders and run decision
  const onApplyCounterfactual = (cf) => {
    const base = selectedIndex != null ? rows[selectedIndex] : null
    if (!base || !cf?.decision_factor) return
    const next = { ...base }
    if (cf.current_value !== undefined && cf.change_direction === 'improve' && typeof cf.current_value === 'number') {
      const r = featureRanges[cf.decision_factor]
      if (r) next[cf.decision_factor] = Math.min(r.max ?? 100, cf.current_value * 1.1)
    } else if (cf.current_value !== undefined && typeof cf.current_value === 'number') {
      next[cf.decision_factor] = cf.current_value
    }
    setWhatIfRow(next)
    setExplanationMode('whatif')
    setTimeout(() => runDecision(next, true), 100)
  }

  const addBookmark = () => {
    if (selectedIndex == null || !result || !activeDataset) return
    const row = rows[selectedIndex]
    const entry = {
      id: `bm_${Date.now()}`,
      datasetId: activeDataset.id,
      datasetName: activeDataset.name,
      rowIndex: selectedIndex,
      row: { ...row },
      result: { ...result },
      savedAt: new Date().toISOString(),
    }
    setBookmarks((prev) => [entry, ...prev].slice(0, 50))
  }

  const removeBookmark = (id) => {
    setBookmarks((prev) => prev.filter((b) => b.id !== id))
  }

  const loadBookmark = async (entry) => {
    setShowSavedCases(false)
    const sameDataset = activeDataset?.id === entry.datasetId
    const rowInRange = entry.rowIndex != null && rows[entry.rowIndex]

    if (sameDataset && rowInRange) {
      setSelectedIndex(entry.rowIndex)
      setResult(entry.result)
      setWhatIfRow(null)
      setBaselineResult(entry.result)
      explanationSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      return
    }

    setPendingBookmark(entry)
    try {
      if (entry.datasetId?.startsWith?.('upload_')) {
        await selectRecentUpload(entry.datasetId)
      } else {
        await selectCatalogDataset(entry.datasetId)
      }
    } catch (e) {
      setError(e.message)
      setPendingBookmark(null)
    }
  }

  const handleBulkExport = () => {
    const preds = {}
    Object.entries(rowPredictions).forEach(([k, v]) => {
      preds[k] = { decision: v.decision, confidence: v.confidence }
    })
    exportBulk({ columns, rows: rows.slice(0, 100), predictions: preds }).catch(setError)
  }

  const finishTour = () => {
    setTourStep(null)
    try { localStorage.setItem('prism_tour_done', 'true') } catch (_) {}
  }

  const handleModeChange = (mode) => {
    if (!DEMO_MODE) return
    const from = explanationMode
    setExplanationMode(mode)
    trackInteraction('change_mode', { from, to: mode })
    trackSectionEnter(mode)
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
  const isStaticMode = false
  const isMinimalMode = false
  const trustCal = result?.trust_calibration

  // Persist theme and apply class
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('prism_theme', theme)
  }, [theme])

  // Persist bookmarks
  useEffect(() => {
    try { localStorage.setItem('prism_bookmarks', JSON.stringify(bookmarks)) } catch (_) {}
  }, [bookmarks])

  // Apply pending bookmark after dataset loads
  useEffect(() => {
    if (!pendingBookmark || activeDataset?.id !== pendingBookmark.datasetId || rows.length === 0) return
    const idx = Math.min(pendingBookmark.rowIndex ?? 0, rows.length - 1)
    setSelectedIndex(idx)
    setResult(pendingBookmark.result)
    setWhatIfRow(null)
    setBaselineResult(pendingBookmark.result)
    setPendingBookmark(null)
    requestAnimationFrame(() => {
      explanationSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }, [pendingBookmark, activeDataset?.id, rows.length])

  // Keyboard navigation: ArrowUp/Down change selection, Enter runs decision
  const selectedIndexRef = useRef(selectedIndex)
  const rowsRef = useRef(rows)
  const onSelectRowRef = useRef(onSelectRow)
  selectedIndexRef.current = selectedIndex
  rowsRef.current = rows
  onSelectRowRef.current = onSelectRow
  useEffect(() => {
    const onKeyDown = (e) => {
      const r = rowsRef.current
      if (r.length === 0) return
      if (e.target?.closest?.('input, select, textarea')) return
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex((prev) => (prev == null ? 0 : Math.min(prev + 1, r.length - 1)))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex((prev) => (prev == null ? r.length - 1 : Math.max(0, prev - 1)))
      } else if (e.key === 'Enter') {
        const idx = selectedIndexRef.current
        if (idx != null && r[idx]) {
          e.preventDefault()
          onSelectRowRef.current(r[idx], idx, false)
        }
      } else if (e.key === 'Escape') {
        setCompareIndex(null)
        setShowSavedCases(false)
        setTourStep(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [rows.length])

  // Demo mode: load dataset catalog on mount
  useEffect(() => {
    loadDatasetCatalog()
  }, [loadDatasetCatalog])

  // ============== RENDER: MAIN APP (Demo — no study, no participants, no questionnaires) ==============
  return (
    <div className={`app theme-${theme}`} role="application" aria-label="PRISM Explainable AI">
      <header className="header">
        <div className="header-row">
          <div>
            <h1 id="app-title">PRISM</h1>
            <p className="tagline">PRISM — Human-centred Explainable AI (Demo)</p>
          </div>
          <div className="header-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={() => setShowSavedCases((s) => !s)}
              aria-label="Saved cases"
              title="Saved cases"
            >
              {bookmarks.length > 0 ? `★ ${bookmarks.length}` : '☆ Saved'}
            </button>
            <button
              type="button"
              role="switch"
              aria-checked={theme === 'light'}
              className="theme-toggle"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
            >
              <span className="theme-toggle-track">
                <span className="theme-toggle-thumb" />
              </span>
            </button>
            {!localStorage.getItem('prism_tour_done') && (
              <button type="button" className="icon-btn" onClick={() => setTourStep(0)} aria-label="Start tour">
                ?
              </button>
            )}
          </div>
        </div>
        {rows.length > 0 && (
          <div className="progress-mini">
            Reviewed: {selectedIndex !== null ? selectedIndex + 1 : 0} / {rows.length}
            {compareIndex != null && (
              <span className="compare-badge">Comparing row {compareIndex + 1}</span>
            )}
          </div>
        )}
      </header>

      {/* Onboarding tour */}
      {tourStep != null && (
        <div className="tour-overlay" role="dialog" aria-labelledby="tour-title">
          <div className="tour-modal">
            <h2 id="tour-title">Welcome to PRISM</h2>
            {tourStep === 0 && (
              <>
                <p>PRISM helps you understand AI decisions with plain-language explanations, SHAP factors, and what-if scenarios.</p>
                <p><strong>Steps:</strong> Select a dataset → Pick a row → See the decision and explanation → Try What-If sliders.</p>
                <p className="muted">Press Escape to skip. Use ↑↓ and Enter to navigate the table with the keyboard.</p>
              </>
            )}
            {tourStep === 1 && <p>Choose a dataset from the catalog or upload your own CSV. Then select a row in the table.</p>}
            {tourStep === 2 && <p>Click a row to see PRISM&apos;s decision, confidence, and explanation. Use Ctrl/Cmd+click on another row to compare two rows side-by-side.</p>}
            {tourStep === 3 && <p>Switch between Plain Language, Technical (SHAP), and What-If modes. Use &quot;Try this&quot; on suggestions to apply them to the sliders.</p>}
            {tourStep === 4 && <p>You can bookmark cases, export CSV/PDF, and filter or search the table. Enjoy exploring!</p>}
            <div className="tour-actions">
              {tourStep < 4 ? (
                <button type="button" className="btn primary" onClick={() => setTourStep((s) => s + 1)}>Next</button>
              ) : (
                <button type="button" className="btn primary" onClick={finishTour}>Got it</button>
              )}
              <button type="button" className="btn secondary" onClick={finishTour}>Skip tour</button>
            </div>
          </div>
        </div>
      )}

      {/* Saved cases panel */}
      {showSavedCases && (
        <div className="saved-panel" role="dialog" aria-label="Saved cases">
          <div className="saved-panel-inner">
            <h3>Saved cases</h3>
            {bookmarks.length === 0 ? (
              <p className="muted">No saved cases. Select a row and use &quot;Save case&quot; to bookmark.</p>
            ) : (
              <ul className="saved-list">
                {bookmarks.map((b) => (
                  <li key={b.id}>
                    <button type="button" className="saved-item" onClick={() => loadBookmark(b)}>
                      {b.datasetName} — Row {b.rowIndex + 1} — {b.result?.decision?.decision ?? '—'}
                    </button>
                    <button type="button" className="saved-remove" onClick={(e) => { e.stopPropagation(); removeBookmark(b.id) }} aria-label="Remove">×</button>
                  </li>
                ))}
              </ul>
            )}
            <button type="button" className="btn secondary" onClick={() => setShowSavedCases(false)}>Close</button>
          </div>
        </div>
      )}

      {error && (
        <div className="banner error" role="alert">
          {error}
          <button type="button" onClick={() => setError('')} aria-label="Dismiss">×</button>
        </div>
      )}
      
      {/* Training Wizard Modal */}
      {showTrainingWizard && trainingUpload && (
        <div className="wizard-overlay">
          <div className="wizard-modal">
            <div className="wizard-header">
              <h2>Configure Dataset for PRISM</h2>
              <p className="wizard-subtitle">
                Set up <strong>{trainingUpload.filename}</strong> for AI-powered decisions and explanations
              </p>
            </div>
            
            {trainingError && (
              <div className="wizard-error">{trainingError}</div>
            )}
            
            {trainingStatus?.status === 'completed' ? (
              <div className="wizard-success">
                <div className="success-icon">✓</div>
                <h3>Training Complete!</h3>
                <p>Model accuracy: <strong>{((trainingStatus.accuracy || 0) * 100).toFixed(1)}%</strong></p>
                <p className="muted">You can now get full PRISM explanations for this dataset.</p>
              </div>
            ) : trainingStatus?.status === 'training' || trainingStatus?.status === 'configuring' ? (
              <div className="wizard-progress">
                <div className="progress-spinner"></div>
                <p>Training model... {trainingStatus.progress}%</p>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${trainingStatus.progress}%` }}></div>
                </div>
              </div>
            ) : (
              <>
                {/* Step 1: Target Column */}
                <div className="wizard-section">
                  <h3>1. Select Target Column</h3>
                  <p className="muted">Choose the column that contains the outcome you want to predict.</p>
                  
                  <div className="wizard-columns">
                    {trainingAnalysis?.columns?.filter(c => c.is_target_candidate).map(col => (
                      <label 
                        key={col.name}
                        className={`wizard-column-option ${trainingConfig.targetCol === col.name ? 'selected' : ''}`}
                      >
                        <input
                          type="radio"
                          name="targetCol"
                          value={col.name}
                          checked={trainingConfig.targetCol === col.name}
                          onChange={() => handleTargetColChange(col.name)}
                        />
                        <span className="column-name">{col.suggested_label || col.name}</span>
                        <span className="column-meta">
                          {col.unique_count} values • {col.dtype}
                        </span>
                      </label>
                    ))}
                    
                    {/* Show other columns */}
                    <details className="other-columns">
                      <summary>Other columns ({trainingAnalysis?.columns?.filter(c => !c.is_target_candidate).length})</summary>
                      {trainingAnalysis?.columns?.filter(c => !c.is_target_candidate).map(col => (
                        <label 
                          key={col.name}
                          className={`wizard-column-option ${trainingConfig.targetCol === col.name ? 'selected' : ''}`}
                        >
                          <input
                            type="radio"
                            name="targetCol"
                            value={col.name}
                            checked={trainingConfig.targetCol === col.name}
                            onChange={() => handleTargetColChange(col.name)}
                          />
                          <span className="column-name">{col.suggested_label || col.name}</span>
                          <span className="column-meta">
                            {col.unique_count} values • {col.dtype}
                          </span>
                        </label>
                      ))}
                    </details>
                  </div>
                </div>
                
                {/* Step 2: Positive/Negative Values */}
                {targetInfo && targetInfo.unique_values && (
                  <div className="wizard-section">
                    <h3>2. Define Outcome Labels</h3>
                    <p className="muted">Select which value represents the "positive" outcome (e.g., Approved, Pass, Yes).</p>
                    
                    <div className="wizard-values">
                      <div className="value-group">
                        <label>Positive Outcome</label>
                        <select 
                          value={trainingConfig.positiveValue ?? ''}
                          onChange={(e) => setTrainingConfig(prev => ({ 
                            ...prev, 
                            positiveValue: e.target.value 
                          }))}
                        >
                          <option value="">Select value...</option>
                          {targetInfo.unique_values.map(v => (
                            <option key={String(v)} value={v}>{String(v)}</option>
                          ))}
                        </select>
                        <input
                          type="text"
                          placeholder="Label (e.g., Approved)"
                          value={trainingConfig.positiveLabel}
                          onChange={(e) => setTrainingConfig(prev => ({ ...prev, positiveLabel: e.target.value }))}
                        />
                      </div>
                      
                      <div className="value-group">
                        <label>Negative Outcome</label>
                        <select 
                          value={trainingConfig.negativeValue ?? ''}
                          onChange={(e) => setTrainingConfig(prev => ({ 
                            ...prev, 
                            negativeValue: e.target.value 
                          }))}
                        >
                          <option value="">Select value...</option>
                          {targetInfo.unique_values.map(v => (
                            <option key={String(v)} value={v}>{String(v)}</option>
                          ))}
                        </select>
                        <input
                          type="text"
                          placeholder="Label (e.g., Rejected)"
                          value={trainingConfig.negativeLabel}
                          onChange={(e) => setTrainingConfig(prev => ({ ...prev, negativeLabel: e.target.value }))}
                        />
                      </div>
                    </div>
                    
                    {targetInfo.value_counts && (
                      <div className="value-distribution">
                        <span className="muted">Distribution: </span>
                        {Object.entries(targetInfo.value_counts).map(([val, count]) => (
                          <span key={val} className="dist-item">
                            {String(val)}: {count}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                
                {/* Step 3: Name */}
                <div className="wizard-section">
                  <h3>3. Dataset Name (optional)</h3>
                  <input
                    type="text"
                    placeholder="My Custom Dataset"
                    value={trainingConfig.name}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, name: e.target.value }))}
                    className="wizard-name-input"
                  />
                </div>
                
                {/* Analysis warnings */}
                {trainingAnalysis?.warnings?.length > 0 && (
                  <div className="wizard-warnings">
                    {trainingAnalysis.warnings.map((w, i) => (
                      <div key={i} className="warning-item">⚠️ {w}</div>
                    ))}
                  </div>
                )}
              </>
            )}
            
            <div className="wizard-actions">
              {trainingStatus?.status !== 'completed' && trainingStatus?.status !== 'training' && (
                <>
                  <button type="button" className="wizard-btn secondary" onClick={skipTraining}>
                    Skip — Explore Only
                  </button>
                  <button 
                    type="button" 
                    className="wizard-btn primary" 
                    onClick={startTraining}
                    disabled={!trainingConfig.targetCol || trainingConfig.positiveValue === null}
                  >
                    Train Model
                  </button>
                </>
              )}
              {trainingStatus?.status === 'completed' && (
                <button type="button" className="wizard-btn primary" onClick={() => setShowTrainingWizard(false)}>
                  Continue to PRISM
                </button>
              )}
            </div>
            
            <button 
              type="button" 
              className="wizard-close" 
              onClick={() => setShowTrainingWizard(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
        </div>
      )}

      <section className="section">
        <h2>Dataset</h2>
        
        {/* Active dataset info with context */}
        {activeDataset && (
          <div className="active-dataset-section">
            <div className="active-dataset-banner">
              {activeDataset.domain && (
                <span className={`domain-tag domain-${activeDataset.domain?.toLowerCase()}`}>
                  {activeDataset.domain}
                </span>
              )}
              <span className="dataset-name">{activeDataset.name}</span>
              {activeDataset.accuracy && (
                <span className="accuracy-indicator" title="Model Accuracy">
                  {(activeDataset.accuracy * 100).toFixed(0)}% accuracy
                </span>
              )}
              <button type="button" className="change-dataset-btn" onClick={() => { loadDatasetCatalog(); setShowDatasetPicker(true) }}>
                Change Dataset
              </button>
            </div>
            {/* Dataset context description */}
            <div className="dataset-context">
              <p className="context-description">
                <strong>What you're examining:</strong>{' '}
                {activeDataset.description || `Analyzing ${activeDataset.name} data`}.{' '}
                Select a row from the table below to see PRISM's prediction, explanation, and stability analysis.
                {activeDataset.positiveLabel && activeDataset.negativeLabel && (
                  <span className="outcome-labels">
                    {' '}Outcomes: <span className="positive-label">{activeDataset.positiveLabel}</span> or <span className="negative-label">{activeDataset.negativeLabel}</span>.
                  </span>
                )}
              </p>
            </div>
            {/* Dataset summary (class balance, feature count) */}
            {datasetSummary && (
              <div className="dataset-summary" role="region" aria-label="Dataset summary">
                <span className="summary-item">{datasetSummary.row_count?.toLocaleString() ?? 0} rows</span>
                <span className="summary-item">{datasetSummary.feature_count ?? columns.length} features</span>
                {datasetSummary.positive_count != null && datasetSummary.negative_count != null && (
                  <>
                    <span className="summary-item positive">
                      {activeDataset?.positiveLabel || 'Positive'}: {datasetSummary.positive_count}
                    </span>
                    <span className="summary-item negative">
                      {activeDataset?.negativeLabel || 'Negative'}: {datasetSummary.negative_count}
                    </span>
                  </>
                )}
                {datasetSummary.class_balance && !datasetSummary.positive_count && (
                  <span className="summary-item">
                    Classes: {Object.entries(datasetSummary.class_balance).map(([k, v]) => `${k}: ${v}`).join(', ')}
                  </span>
                )}
              </div>
            )}
            {/* Global explainability (model-level feature importance) */}
            {globalExplainability?.feature_names?.length > 0 && (
              <div className="global-explainability" role="region" aria-label="Model overview">
                <h4>What drives this model (overall)</h4>
                <p className="muted">Mean impact of each factor across a sample of rows (global SHAP).</p>
                <div className="global-shap-bars">
                  {globalExplainability.feature_names.slice(0, 10).map((name, i) => {
                    const label = globalExplainability.feature_labels?.[name] || name
                    const val = globalExplainability.mean_abs_shap?.[i] ?? 0
                    const maxVal = Math.max(...(globalExplainability.mean_abs_shap || [1]), 0.01)
                    return (
                      <div key={name} className="global-shap-row">
                        <span className="global-shap-label" title={name}>{String(label).slice(0, 24)}{String(label).length > 24 ? '…' : ''}</span>
                        <div className="global-shap-bar-bg">
                          <div className="global-shap-bar-fill" style={{ width: `${(val / maxVal) * 100}%` }} />
                        </div>
                        <span className="global-shap-val">{val.toFixed(3)}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Dataset picker actions */}
        <div className="dataset-actions">
          <button 
            type="button" 
            onClick={() => { loadDatasetCatalog(); setShowDatasetPicker(!showDatasetPicker) }} 
            disabled={loading}
            className={showDatasetPicker ? 'active' : ''}
          >
            {rows.length === 0 ? 'Select a Dataset' : 'Browse Datasets'}
          </button>
          <label className="button-like">
            Upload CSV
            <input type="file" accept=".csv" onChange={onUpload} disabled={loading} hidden />
          </label>
        </div>
        
        {/* Dataset picker dropdown */}
        {showDatasetPicker && (
          <div className="dataset-picker">
            {/* Group datasets by domain */}
            {(() => {
              const domainGroups = {
                'Finance': { color: '#22c55e', icon: '💰', datasets: [] },
                'Healthcare': { color: '#ef4444', icon: '🏥', datasets: [] },
                'Education': { color: '#3b82f6', icon: '📚', datasets: [] },
                'Employment': { color: '#8b5cf6', icon: '💼', datasets: [] },
                'Insurance': { color: '#f59e0b', icon: '🛡️', datasets: [] },
                'Legal': { color: '#64748b', icon: '⚖️', datasets: [] },
              }
              datasetCatalog.forEach(ds => {
                const domain = ds.domain || 'Other'
                if (!domainGroups[domain]) {
                  domainGroups[domain] = { color: '#888', icon: '📊', datasets: [] }
                }
                domainGroups[domain].datasets.push(ds)
              })
              
              return Object.entries(domainGroups).map(([domain, group]) => (
                group.datasets.length > 0 && (
                  <div className="picker-section" key={domain}>
                    <h4 className="domain-header" style={{ borderLeftColor: group.color }}>
                      <span className="domain-icon">{group.icon}</span>
                      {domain}
                      <span className="domain-count">{group.datasets.length} datasets</span>
                    </h4>
                    <div className="dataset-grid">
                      {group.datasets.map(ds => (
                        <div 
                          key={ds.id} 
                          className={`dataset-card ${ds.model_compatible ? 'compatible' : 'exploration'}`}
                          onClick={() => selectCatalogDataset(ds.id)}
                          style={{ '--domain-color': group.color }}
                        >
                          <div className="dataset-card-header">
                            <span className="dataset-title">{ds.name}</span>
                            {ds.accuracy && (
                              <span className="accuracy-badge">{(ds.accuracy * 100).toFixed(0)}%</span>
                            )}
                          </div>
                          <p className="dataset-desc">{ds.description}</p>
                          <div className="dataset-meta">
                            <span>{ds.rows?.toLocaleString()} rows</span>
                            <span>{ds.features} features</span>
                          </div>
                          <div className="dataset-source">{ds.source}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              ))
            })()}
            
            {/* Recent uploads */}
            {recentUploads.length > 0 && (
              <div className="picker-section">
                <h4 className="domain-header" style={{ borderLeftColor: '#888' }}>
                  <span className="domain-icon">📁</span>
                  Recent Uploads
                  <span className="domain-count">{recentUploads.length} files</span>
                </h4>
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
            {/* Table header with search and filter */}
            <div className="table-header">
              <p className="muted">
                {predictingBatch ? 'Predicting outcomes...' : 'Select a row to see full explanation. Ctrl/Cmd+click to compare two rows.'}
                {' '}Showing up to 30 rows.
              </p>
              <div className="table-controls">
                <input
                  type="search"
                  className="table-search"
                  placeholder="Search in table…"
                  value={tableSearch}
                  onChange={(e) => setTableSearch(e.target.value)}
                  aria-label="Search rows"
                />
                {Object.keys(rowPredictions).length > 0 && (
                  <div className="prediction-filter">
                    <label htmlFor="pred-filter">Filter by outcome:</label>
                    <select id="pred-filter" value={predictionFilter} onChange={(e) => setPredictionFilter(e.target.value)}>
                      <option value="all">All ({rows.length})</option>
                      <option value="positive">
                        {activeDataset?.positiveLabel || 'Positive'} ({
                          Object.values(rowPredictions).filter(p =>
                            p.decision === p.positiveLabel || p.decision === '+'
                          ).length
                        })
                      </option>
                      <option value="negative">
                        {activeDataset?.negativeLabel || 'Negative'} ({
                          Object.values(rowPredictions).filter(p =>
                            p.decision === p.negativeLabel || p.decision === '-'
                          ).length
                        })
                      </option>
                    </select>
                  </div>
                )}
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {Object.keys(rowPredictions).length > 0 && <th className="prediction-col">Prediction</th>}
                  {columns.slice(0, 5).map((c) => <th key={c}>{c}</th>)}
                  <th>…</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 100)
                  .map((row, i) => ({ row, i, prediction: rowPredictions[i] }))
                  .filter(({ row, prediction }) => {
                    if (predictionFilter !== 'all' && prediction) {
                      const isPositive = prediction.decision === prediction.positiveLabel || prediction.decision === '+'
                      if (predictionFilter === 'positive' && !isPositive) return false
                      if (predictionFilter === 'negative' && isPositive) return false
                    }
                    if (!tableSearch.trim()) return true
                    const q = tableSearch.toLowerCase()
                    const searchCols = columns.slice(0, 8)
                    return searchCols.some((c) => String(row[c] ?? '').toLowerCase().includes(q))
                  })
                  .slice(0, 30)
                  .map(({ row, i, prediction }) => {
                    const isPositive = prediction && (prediction.decision === prediction.positiveLabel || prediction.decision === '+')
                    const predClass = prediction ? (isPositive ? 'pred-positive' : 'pred-negative') : ''
                    return (
                      <tr
                        key={i}
                        className={`${selectedIndex === i ? 'selected' : ''} ${compareIndex === i ? 'compare' : ''} ${predClass}`}
                        onClick={(e) => onSelectRow(row, i, e.ctrlKey || e.metaKey)}
                        tabIndex={0}
                        onKeyDown={(e) => { if (e.key === 'Enter') onSelectRow(row, i, e.ctrlKey || e.metaKey) }}
                      >
                        <td>{i + 1}</td>
                        {Object.keys(rowPredictions).length > 0 && (
                          <td className="prediction-cell">
                            {prediction ? (
                              <span className={`prediction-badge ${isPositive ? 'positive' : 'negative'}`}>
                                {prediction.decision === '+' ? activeDataset?.positiveLabel || 'Approved' 
                                  : prediction.decision === '-' ? activeDataset?.negativeLabel || 'Rejected' 
                                  : prediction.decision}
                                <span className="pred-conf">{((prediction.confidence ?? 0) * 100).toFixed(0)}%</span>
                              </span>
                            ) : (
                              <span className="prediction-badge pending">...</span>
                            )}
                          </td>
                        )}
                        {columns.slice(0, 5).map((c) => <td key={c}>{String(row[c] ?? '')}</td>)}
                        <td>…</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
            {predictingBatch && (
              <div className="batch-prediction-loading">
                <span className="spinner"></span>
                <span>Computing predictions for all rows...</span>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-dataset-prompt">
            <p>Select a dataset to begin exploring PRISM explanations.</p>
            <p className="muted">Choose from 11 pre-loaded datasets across Finance, Healthcare, Education, Employment, Insurance, and Legal domains — or upload your own CSV.</p>
            <button 
              type="button" 
              className="select-dataset-btn"
              onClick={() => { loadDatasetCatalog(); setShowDatasetPicker(true) }}
            >
              Select a Dataset
            </button>
          </div>
        )}
      </section>

      {(selectedIndex != null || result) && (
      <section ref={explanationSectionRef} className="section" onMouseEnter={() => trackSectionEnter('explanation')}>
        <h2>Explanation</h2>
        
        {/* Mode toggle - interactive only */}
        {DEMO_MODE && (
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
          </div>
        )}

        {deciding && <p className="muted">Computing decision…</p>}
        {decidingB && compareIndex != null && <p className="muted">Computing second decision…</p>}

        {result && !deciding && (
          <>
            {/* Compare mode: two decision cards side-by-side */}
            {compareIndex != null && rows[compareIndex] ? (
              <div className="compare-grid" role="region" aria-label="Compare two rows">
                <div className="compare-column">
                  <h3 className="compare-label">Row {selectedIndex + 1}</h3>
                  <div className="decision-card decision-primary">
                    <div className="decision-header"><h3>PRISM Decision</h3></div>
                    <div className="decision-content">
                      <div className="decision-outcome-row">
                        <span className={`decision-badge ${(result.decision?.decision === result.decision?.positive_label || result.decision?.decision === '+') ? 'positive' : 'negative'}`}>
                          {result.decision?.decision === '+' ? (activeDataset?.positiveLabel || 'Approved') : result.decision?.decision === '-' ? (activeDataset?.negativeLabel || 'Rejected') : result.decision?.decision}
                        </span>
                        <strong className="confidence-value">{((result.decision?.confidence ?? 0) * 100).toFixed(1)}%</strong>
                      </div>
                    </div>
                  </div>
                  {result.explanation_layer?.bullets?.length > 0 && (
                    <div className="explanation-layer-card">
                      <h4>Key factors</h4>
                      <ul className="plain-language-bullets">{result.explanation_layer.bullets.slice(0, 3).map((b, i) => <li key={i}>{b}</li>)}</ul>
                    </div>
                  )}
                </div>
                <div className="compare-column">
                  <h3 className="compare-label">Row {compareIndex + 1}</h3>
                  {resultB && !decidingB ? (
                    <>
                      <div className="decision-card decision-primary">
                        <div className="decision-header"><h3>PRISM Decision</h3></div>
                        <div className="decision-content">
                          <div className="decision-outcome-row">
                            <span className={`decision-badge ${(resultB.decision?.decision === resultB.decision?.positive_label || resultB.decision?.decision === '+') ? 'positive' : 'negative'}`}>
                              {resultB.decision?.decision === '+' ? (activeDataset?.positiveLabel || 'Approved') : resultB.decision?.decision === '-' ? (activeDataset?.negativeLabel || 'Rejected') : resultB.decision?.decision}
                            </span>
                            <strong className="confidence-value">{((resultB.decision?.confidence ?? 0) * 100).toFixed(1)}%</strong>
                          </div>
                        </div>
                      </div>
                      {resultB.explanation_layer?.bullets?.length > 0 && (
                        <div className="explanation-layer-card">
                          <h4>Key factors</h4>
                          <ul className="plain-language-bullets">{resultB.explanation_layer.bullets.slice(0, 3).map((b, i) => <li key={i}>{b}</li>)}</ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="decision-card decision-primary"><p className="muted">Loading…</p></div>
                  )}
                </div>
                <div className="compare-clear-wrap">
                  <button type="button" className="btn secondary" onClick={() => { setCompareIndex(null); setResultB(null) }}>Clear compare</button>
                </div>
              </div>
            ) : (
            <>
            {/* ========== PRIMARY: DECISION CARD (most prominent) ========== */}
            <div className="decision-card decision-primary" onMouseEnter={handleHoverStart} onMouseLeave={() => handleHoverEnd('decision')}>
              <div className="decision-header">
                <h3>PRISM Decision</h3>
                <div className="decision-header-badges">
                  {result.explanation_fidelity && (
                    <span
                      className={`fidelity-badge ${result.explanation_fidelity.prediction_match ? 'high' : 'low'}`}
                      title={result.explanation_fidelity.prediction_match ? 'Explanation matches model decision (high fidelity)' : 'Explanation may not fully match model (check technical view)'}
                    >
                      Explanation: {result.explanation_fidelity.prediction_match ? 'High confidence' : 'Lower confidence'}
                    </span>
                  )}
                  {DEMO_MODE && trustCal && (
                    <span className="model-accuracy" title="Historical model accuracy">
                      Model accuracy: {((trustCal.historical_accuracy || 0) * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </div>
              {(() => {
                const dec = result.decision || {}
                const posLabel = dec.positive_label || 'Approved'
                const negLabel = dec.negative_label || 'Rejected'
                const decision = dec.decision || ''
                const isPositive = decision === '+' || decision === posLabel
                const displayDecision = decision === '+' ? 'Approved' : decision === '-' ? 'Rejected' : decision
                const probs = dec.probabilities || {}
                const posProb = probs[posLabel] ?? probs['+'] ?? 0
                const negProb = probs[negLabel] ?? probs['-'] ?? 0
                const confBand = result.uncertainty?.confidence_band || 'medium'
                
                return (
                  <div className="decision-content">
                    <div className="decision-outcome-row">
                      <span className={`decision-badge ${isPositive ? 'positive' : 'negative'}`}>
                        {displayDecision}
                      </span>
                      <div className="confidence-meter">
                        <div className="confidence-label">
                          <span>Confidence</span>
                          <strong className={`confidence-value band-${confBand}`}>{((dec.confidence ?? 0) * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="confidence-bar-bg">
                          <div 
                            className={`confidence-bar-fill band-${confBand}`} 
                            style={{ width: `${(dec.confidence ?? 0) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                    {!isMinimalMode && (
                      <div className="probability-breakdown">
                        <span className="prob-item positive">P({posLabel}): {(posProb * 100).toFixed(1)}%</span>
                        <span className="prob-item negative">P({negLabel}): {(negProb * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {/* Stability warning inline */}
                    {result.uncertainty?.warning && (
                      <div className="stability-warning">
                        <span className="warning-icon">⚠️</span>
                        <span>{result.uncertainty.warning}</span>
                      </div>
                    )}
                  </div>
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
            {DEMO_MODE && (
              <>
                {/* Plain language explanation */}
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
                      {result.counterfactual_preview.map((p, i) => (
                        <li key={i}>
                          {p.suggestion}
                          {p.decision_factor && (p.current_value !== undefined || p.change_direction) && (
                            <button
                              type="button"
                              className="try-this-btn"
                              onClick={() => onApplyCounterfactual(p)}
                            >
                              Try this
                            </button>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* What-if sliders - dynamic based on dataset */}
                {explanationMode === 'whatif' && Object.keys(featureRanges).length > 0 && activeRow && (
                  <div className="whatif-card">
                    <div className="whatif-header">
                      <h3>What-If Scenarios</h3>
                      <label className="comparison-toggle">
                        <input 
                          type="checkbox" 
                          checked={showComparison} 
                          onChange={(e) => setShowComparison(e.target.checked)}
                        />
                        <span>Compare with original</span>
                      </label>
                    </div>
                    <p className="muted">Adjust values and see how the decision changes.</p>
                    
                    {/* Comparison View */}
                    {showComparison && baselineResult && result && whatIfRow && (
                      <div className="whatif-comparison">
                        <div className="comparison-column baseline">
                          <div className="comparison-label">Original</div>
                          <div className={`comparison-outcome ${baselineResult.decision?.decision === baselineResult.decision?.positive_label || baselineResult.decision?.decision === '+' ? 'positive' : 'negative'}`}>
                            {baselineResult.decision?.decision === '+' ? 'Approved' : baselineResult.decision?.decision === '-' ? 'Rejected' : baselineResult.decision?.decision}
                          </div>
                          <div className="comparison-confidence">
                            {((baselineResult.decision?.confidence ?? 0) * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="comparison-arrow">→</div>
                        <div className="comparison-column modified">
                          <div className="comparison-label">Modified</div>
                          <div className={`comparison-outcome ${result.decision?.decision === result.decision?.positive_label || result.decision?.decision === '+' ? 'positive' : 'negative'}`}>
                            {result.decision?.decision === '+' ? 'Approved' : result.decision?.decision === '-' ? 'Rejected' : result.decision?.decision}
                          </div>
                          <div className="comparison-confidence">
                            {((result.decision?.confidence ?? 0) * 100).toFixed(1)}%
                          </div>
                        </div>
                        {/* Change indicator */}
                        {baselineResult.decision?.decision !== result.decision?.decision && (
                          <div className="outcome-changed">
                            Decision Changed!
                          </div>
                        )}
                        {baselineResult.decision?.decision === result.decision?.decision && (
                          <div className="confidence-delta">
                            Confidence: {(((result.decision?.confidence ?? 0) - (baselineResult.decision?.confidence ?? 0)) * 100) >= 0 ? '+' : ''}
                            {(((result.decision?.confidence ?? 0) - (baselineResult.decision?.confidence ?? 0)) * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div className="sliders">
                      {Object.keys(featureRanges).slice(0, MAX_WHATIF_SLIDERS).map((f) => {
                        const r = featureRanges[f]
                        const min = r.min ?? 0
                        const max = Math.max(r.max ?? 100, min + 1)
                        const raw = activeRow[f]
                        const originalRaw = selectedIndex != null ? rows[selectedIndex]?.[f] : null
                        const val = typeof raw === 'number' && !Number.isNaN(raw) ? raw : (parseFloat(raw) || min)
                        const originalVal = typeof originalRaw === 'number' && !Number.isNaN(originalRaw) ? originalRaw : (parseFloat(originalRaw) || min)
                        const clamped = Math.min(max, Math.max(min, val))
                        const step = Number.isInteger(min) && Number.isInteger(max) ? 1 : (max - min) / 100
                        const isModified = whatIfRow && whatIfRow[f] !== undefined && Math.abs(val - originalVal) > 0.01
                        return (
                          <div key={f} className={`slider-row ${isModified ? 'modified' : ''}`}>
                            <label>
                              {r.label || f}: <strong>{typeof clamped === 'number' ? clamped.toFixed(1) : clamped}</strong>
                              {isModified && showComparison && (
                                <span className="original-value">(was {originalVal.toFixed(1)})</span>
                              )}
                            </label>
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
                    {Object.keys(featureRanges).length > MAX_WHATIF_SLIDERS && (
                      <p className="muted" style={{marginTop: 'var(--space-sm)', fontSize: '12px'}}>
                        Showing {MAX_WHATIF_SLIDERS} of {Object.keys(featureRanges).length} numeric factors for clarity.
                      </p>
                    )}
                    <div className="whatif-actions">
                      <button type="button" className="whatif-apply" onClick={onApplyWhatIf} disabled={!whatIfRow}>
                        Update Decision
                      </button>
                      <button type="button" className="whatif-reset" onClick={onResetToBaseline} disabled={!whatIfRow}>
                        Reset to Original
                      </button>
                    </div>
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
                {Object.keys(rowPredictions).length > 0 && (
                  <button type="button" onClick={handleBulkExport}>Export all (bulk CSV)</button>
                )}
                <button type="button" className="save-case-btn" onClick={addBookmark} disabled={!result || selectedIndex == null}>
                  Save case
                </button>
              </div>
            )}
            </>
            )}
          </>
        )}
      </section>
      )}

      <footer className="footer">
        <p>PRISM — Human-centred Explainable AI</p>
      </footer>
    </div>
  )
}
