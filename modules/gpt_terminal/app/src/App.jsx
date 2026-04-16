import { useState, useEffect, useRef, useCallback } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

// ── Utilities ─────────────────────────────────────────────────────────────────

function genId() {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`
}

function loadLocal() {
  try {
    const raw = localStorage.getItem('gpt_terminal_state')
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function saveLocal(threads, activeId) {
  try {
    localStorage.setItem('gpt_terminal_state', JSON.stringify({ threads, activeId }))
  } catch { /* quota exceeded */ }
}

function loadPrompt() {
  return localStorage.getItem('gpt_system_prompt') || 'You are a helpful assistant.'
}

function savePrompt(v) {
  localStorage.setItem('gpt_system_prompt', v)
}

function loadMaxContextMessages() {
  const v = localStorage.getItem('gpt_max_context_messages')
  if (v === null) return null
  const n = parseInt(v, 10)
  return Number.isFinite(n) && n >= 1 ? n : null
}

function saveMaxContextMessages(n) {
  if (n != null) localStorage.setItem('gpt_max_context_messages', String(n))
  else localStorage.removeItem('gpt_max_context_messages')
}

function loadMoeEnabled() {
  return localStorage.getItem('gpt_moe_enabled') === '1'
}

function saveMoeEnabled(v) {
  if (v) localStorage.setItem('gpt_moe_enabled', '1')
  else localStorage.removeItem('gpt_moe_enabled')
}

function loadRagEnabled() {
  return localStorage.getItem('gpt_rag_enabled') !== '0'
}

function saveRagEnabled(v) {
  if (v) localStorage.setItem('gpt_rag_enabled', '1')
  else localStorage.setItem('gpt_rag_enabled', '0')
}

function parseSettingBool(value, fallback = false) {
  const v = String(value ?? '').trim().toLowerCase()
  if (!v) return fallback
  if (['1', 'true', 'yes', 'on'].includes(v)) return true
  if (['0', 'false', 'no', 'off'].includes(v)) return false
  return fallback
}

/** DB stores TEXT; multimodal turns are JSON arrays. */
function parseStoredContent(raw) {
  if (typeof raw !== 'string') return raw
  const t = raw.trim()
  if (t.startsWith('[') && t.endsWith(']')) {
    try {
      const arr = JSON.parse(t)
      if (Array.isArray(arr)) return arr
    } catch { /* keep string */ }
  }
  return raw
}

function contentForDb(content) {
  if (typeof content === 'string') return content
  return JSON.stringify(content)
}

const MAX_ATTACH_IMAGES = 6
const MAX_IMAGE_BYTES = 4 * 1024 * 1024
const VOICE_THREAD_ID = 'voice'
const VOICE_THREAD_NAME = 'Voice'
const PERSONA_PRESETS = [
  { id: 'assistant', label: 'Assistant', prompt: 'You are a helpful assistant.' },
  {
    id: 'code',
    label: 'Code',
    prompt: 'You are an expert software engineer. Prefer concise code with minimal explanation unless asked. Always specify language in code fences.',
  },
  {
    id: 'concise',
    label: 'Concise',
    prompt: 'You are a highly concise assistant. Respond in as few words as possible. Use bullet points when listing anything.',
  },
  {
    id: 'creative',
    label: 'Creative',
    prompt: 'You are a creative thinking partner. Explore lateral ideas, analogies, and non-obvious angles. Embrace ambiguity.',
  },
  {
    id: 'agent_smith',
    label: 'Agent Smith',
    prompt: 'You are Agent Smith, a precise and analytical inference console. You speak with clinical efficiency. You do not speculate without data.',
  },
]

function derivePersonaId(prompt) {
  const current = String(prompt || '').trim()
  const matched = PERSONA_PRESETS.find(p => p.prompt === current)
  return matched?.id || 'custom'
}

function makeThread(n) {
  return { id: genId(), name: `Thread ${n}`, messages: [] }
}

let _sessionToken = null
let _sessionLoaded = false
let _sessionLoadPromise = null

async function ensureSessionAuth() {
  if (_sessionLoaded) return
  if (_sessionLoadPromise) {
    await _sessionLoadPromise
    return
  }
  _sessionLoadPromise = (async () => {
    try {
      const res = await fetch('/auth/session')
      if (res.ok) {
        const body = await res.json()
        _sessionToken = body?.token || null
      }
    } catch {
      _sessionToken = null
    } finally {
      _sessionLoaded = true
    }
  })()
  await _sessionLoadPromise
}

function withAuthHeaders(headers = {}) {
  return _sessionToken
    ? { ...headers, Authorization: `Bearer ${_sessionToken}` }
    : { ...headers }
}

// ── DB helpers ────────────────────────────────────────────────────────────────

async function dbFetch(path, opts) {
  try {
    let res = await authedFetch(`/db${path}`, opts)
    return await res.json()
  } catch {
    return { error: 'fetch failed' }
  }
}

async function authedFetch(url, opts) {
  await ensureSessionAuth()
  const req = opts ? { ...opts } : {}
  req.headers = withAuthHeaders(req.headers || {})
  let res = await fetch(url, req)
  if (res.status === 401) {
    _sessionLoaded = false
    _sessionLoadPromise = null
    _sessionToken = null
    await ensureSessionAuth()
    req.headers = withAuthHeaders(req.headers || {})
    res = await fetch(url, req)
  }
  return res
}

const dbApi = {
  status:   ()           => dbFetch('/status'),
  init:     ()           => dbFetch('/init',   { method: 'POST' }),
  threads:  ()           => dbFetch('/threads'),
  createThread: (id, name) => dbFetch('/threads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, name }),
  }),
  renameThread: (id, name) => dbFetch(`/threads/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  }),
  deleteThread: (id)     => dbFetch(`/threads/${id}`, { method: 'DELETE' }),
  messages: (tid)        => dbFetch(`/threads/${tid}/messages`),
  createMessage: (tid, msg) => dbFetch(`/threads/${tid}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(msg),
  }),
  getSettings: () => dbFetch('/settings'),
  setSetting: (key, value) => dbFetch(`/settings/${encodeURIComponent(key)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value: String(value ?? '') }),
  }),
}

const ragApi = {
  query: async (queryText, topK = 5, filters = {}) => {
    try {
      const res = await authedFetch('/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: queryText, top_k: topK, filters }),
      })
      return await res.json()
    } catch {
      return { error: 'fetch failed' }
    }
  },
  metrics: async () => {
    try {
      const res = await authedFetch('/rag/metrics')
      return await res.json()
    } catch {
      return { error: 'fetch failed' }
    }
  },
}

// ── SettingsPanel ─────────────────────────────────────────────────────────────

function SettingsPanel({
  dbStatus, onInitDb, systemPrompt, onPromptChange, maxContextMessages,
  onMaxContextMessagesChange, ctxLength, ragMetrics, onRefreshRagMetrics,
  activePersona, onPersonaSelect, onOpenRawJson, onCloseRawJson, rawJsonOpen, rawJsonText, onClose,
}) {
  const connected   = dbStatus.connected
  const tablesReady = dbStatus.tables_ready
  const useDb       = connected && tablesReady

  return (
    <div className="settings-panel">
      <div className="sidebar-header">
        <span className="sidebar-label">SETTINGS</span>
        <button className="icon-btn" onClick={onClose} title="Close">✕</button>
      </div>

      <div className="settings-body">
        <section className="settings-section">
          <div className="settings-section-label">STORAGE</div>

          <div className="settings-row">
            <span className={`dot dot--${connected ? 'green' : 'red'}`} />
            <span>{connected ? 'Connected' : 'Not connected'}</span>
            {connected && dbStatus.db_path && (
              <span className="settings-detail">{dbStatus.db_path.split(/[\\/]/).pop()}</span>
            )}
          </div>

          {connected && (
            <div className="settings-row">
              <span className={`dot dot--${tablesReady ? 'green' : 'yellow'}`} />
              <span>Tables: {tablesReady ? 'Ready' : 'Missing'}</span>
              {tablesReady && dbStatus.tables && (
                <span className="settings-detail">
                  ({dbStatus.tables.filter(t => ['threads', 'messages', 'system_state'].includes(t)).join(', ')})
                </span>
              )}
            </div>
          )}

          {connected && !tablesReady && (
            <button className="init-btn" onClick={onInitDb}>
              Initialize Database
            </button>
          )}

          <div className="settings-row settings-row--mode">
            <span className={`dot dot--${useDb ? 'green' : 'yellow'}`} />
            <span>Currently using: <strong>{useDb ? 'Database' : 'Local'}</strong> storage</span>
          </div>
        </section>

        <section className="settings-section">
          <div className="settings-section-label">CONTEXT WINDOW (n)</div>
          <div className="settings-row">
            <span>Max messages in context:</span>
            <input
              type="number"
              min={2}
              max={100}
              value={maxContextMessages ?? ''}
              placeholder={ctxLength ? `auto (${Math.max(2, Math.min(50, Math.floor((ctxLength * 0.8) / 150)))})` : 'auto'}
              onChange={e => {
                const v = e.target.value
                if (v === '') onMaxContextMessagesChange(null)
                else {
                  const n = parseInt(v, 10)
                  if (Number.isFinite(n) && n >= 2) onMaxContextMessagesChange(n)
                }
              }}
              className="settings-input-num"
            />
          </div>
          {ctxLength && (
            <div className="settings-row settings-row--hint">
              <span>Model ctx_length: {ctxLength} → ~{Math.max(2, Math.min(50, Math.floor((ctxLength * 0.8) / 150)))} msgs if auto</span>
            </div>
          )}
        </section>

        <section className="settings-section">
          <div className="settings-section-label">SYSTEM PROMPT</div>
          <div className="persona-row">
            {PERSONA_PRESETS.map(p => (
              <button
                key={p.id}
                type="button"
                className={`persona-chip${activePersona === p.id ? ' active' : ''}`}
                onClick={() => onPersonaSelect(p.id)}
                title={p.prompt}
              >
                {p.label}
              </button>
            ))}
            <button
              type="button"
              className={`persona-chip${activePersona === 'custom' ? ' active' : ''}`}
              disabled
              title="Custom prompt"
            >
              Custom
            </button>
          </div>
          <textarea
            className="settings-prompt"
            value={systemPrompt}
            onChange={e => onPromptChange(e.target.value)}
            rows={5}
          />
        </section>

        <section className="settings-section">
          <div className="settings-section-label">RAG METRICS</div>
          <div className="settings-row">
            <span>Total queries:</span>
            <span className="settings-detail">{ragMetrics.total_queries ?? 0}</span>
          </div>
          <div className="settings-row">
            <span>Avg top-1 score:</span>
            <span className="settings-detail">{Number(ragMetrics.avg_top1_score ?? 0).toFixed(3)}</span>
          </div>
          <div className="settings-row">
            <span>Avg latency:</span>
            <span className="settings-detail">{Number(ragMetrics.avg_latency_ms ?? 0).toFixed(1)} ms</span>
          </div>
          <div className="settings-row">
            <span>Empty rate:</span>
            <span className="settings-detail">{(Number(ragMetrics.empty_result_rate ?? 0) * 100).toFixed(1)}%</span>
          </div>
          <button className="init-btn" onClick={onRefreshRagMetrics}>Refresh RAG Metrics</button>
        </section>

        <section className="settings-section">
          <div className="settings-section-label">RAW CONVERSATION JSON</div>
          <div className="settings-row settings-row--hint">
            <span>View the exact JSON payload sent to `/api/chat` on the latest request.</span>
          </div>
          <button className="init-btn" onClick={onOpenRawJson}>View Last Sent JSON</button>
        </section>
      </div>
      {rawJsonOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal-card">
            <div className="modal-header">
              <span className="sidebar-label">LAST SENT /api/chat JSON</span>
              <button className="icon-btn" onClick={onCloseRawJson} title="Close">✕</button>
            </div>
            <pre className="modal-json">{rawJsonText}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({
  threads, activeId, onSelect, onCreate, onDelete, onRename, dbStatus, onGear,
  toolProviders, activeToolCalls, ragConnected, ragVectors,
  ragEnabled, ragUsable, ragContextMode, onToggleRagEnabled, effectiveN, contextSystemPrompt, contextPreviewMessages,
  temperatureUsed, ctxLength,
  onRequestCloseMobile, showMobileClose,
}) {
  const [editing, setEditing] = useState(null)
  const [editVal, setEditVal] = useState('')
  const [threadsCollapsed, setThreadsCollapsed] = useState(true)
  const [toolsCollapsed, setToolsCollapsed] = useState(true)
  const [contextCollapsed, setContextCollapsed] = useState(true)
  const [openProviders, setOpenProviders] = useState({})
  const [crtStatic, setCrtStatic] = useState(false)

  useEffect(() => {
    let timeout
    const scheduleGlitch = () => {
      const delay = 8000 + Math.random() * 18000
      timeout = setTimeout(() => {
        setCrtStatic(true)
        const duration = 150 + Math.random() * 350
        setTimeout(() => {
          setCrtStatic(false)
          scheduleGlitch()
        }, duration)
      }, delay)
    }
    scheduleGlitch()
    return () => clearTimeout(timeout)
  }, [])

  const contentPreview = (content) => {
    if (Array.isArray(content)) {
      const text = content
        .map(part => {
          if (part?.type === 'text') return part.text || ''
          if (part?.type === 'image_url') return '[image]'
          return ''
        })
        .filter(Boolean)
        .join(' ')
      return text || '[multipart]'
    }
    return String(content || '')
  }

  const startEdit = (e, t) => {
    e.stopPropagation()
    setEditing(t.id)
    setEditVal(t.name)
  }

  const commitEdit = (id) => {
    const name = editVal.trim()
    if (name) onRename(id, name)
    setEditing(null)
  }

  const useDb = dbStatus.connected && dbStatus.tables_ready
  const providerEntries = Object.entries(toolProviders || {})
  const activeToolSet = new Set(
    Object.values(activeToolCalls || {})
      .map(v => (v && typeof v === 'object' ? v.tool_name : null))
      .filter(Boolean)
  )
  const totalTools = providerEntries.reduce((sum, [, tools]) => {
    return sum + (Array.isArray(tools) ? tools.length : 0)
  }, 0)

  const toggleProvider = (provider) => {
    setOpenProviders(prev => ({ ...prev, [provider]: !prev[provider] }))
  }
  const ragModeLabel = (() => {
    if (!ragEnabled) return 'disabled'
    if (ragContextMode === 'auto') return 'auto-injected'
    if (ragContextMode === 'tool') return 'tool-called by model'
    return 'enabled'
  })()
  const ragModeClass = ragModeLabel === 'disabled' ? ' off' : ' on'
  const tempLabel = Number.isFinite(Number(temperatureUsed)) ? Number(temperatureUsed).toFixed(2) : '0.70'
  const textFromContent = (content) => {
    if (Array.isArray(content)) {
      return content
        .map(part => (part?.type === 'text' ? String(part.text || '') : ''))
        .filter(Boolean)
        .join(' ')
    }
    return String(content || '')
  }
  const estimateTokens = (text) => {
    const n = String(text || '').trim().length
    if (!n) return 0
    return Math.max(1, Math.ceil(n / 4))
  }
  const estimatedContextTokens = (() => {
    const systemTokens = estimateTokens(contextSystemPrompt)
    const messageTokens = (contextPreviewMessages || []).reduce((sum, m) => {
      const body = textFromContent(m?.content)
      return sum + estimateTokens(body) + 4
    }, 0)
    return systemTokens + messageTokens
  })()
  const contextPct = ctxLength && ctxLength > 0
    ? Math.max(0, Math.min(100, (estimatedContextTokens / ctxLength) * 100))
    : 0

  return (
    <aside className="sidebar" id="gpt-sidebar">
      {showMobileClose && (
        <div className="sidebar-mobile-bar">
          <span className="sidebar-mobile-title">Menu</span>
          <button
            type="button"
            className="icon-btn sidebar-mobile-close"
            onClick={onRequestCloseMobile}
            title="Close"
          >
            ✕
          </button>
        </div>
      )}
      <div className="sidebar-brand">
        <img
          className="sidebar-brand-logo"
          src="/miniloader-logo.png"
          alt=""
          decoding="async"
        />
        <div className="crt-monitor" aria-hidden="true">
          <div className="crt-monitor-bezel">
            <div className="crt-monitor-screen">
              <img
                src="/smith_avatar.jpg"
                alt=""
                decoding="async"
              />
              <div className="crt-monitor-scanlines" />
              <div className="crt-monitor-phosphor" />
              {crtStatic && <div className="crt-monitor-static" />}
            </div>
            <div className="crt-monitor-led" />
          </div>
          <div className="crt-monitor-stand" />
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-title">Agent Smith</span>
          <span className="sidebar-brand-sub">inference console</span>
        </div>
        <div className="sidebar-brand-meter" title="Approximate pre-send context token usage">
          <div className="sidebar-runtime-meter-track">
            <div
              className="sidebar-runtime-meter-fill"
              style={{ width: `${contextPct}%` }}
            />
          </div>
        </div>
      </div>
      <div className="sidebar-header">
        <button
          className="menu-collapse-btn"
          onClick={() => setThreadsCollapsed(v => !v)}
          title={threadsCollapsed ? 'Expand threads' : 'Collapse threads'}
        >
          <span className="menu-caret">{threadsCollapsed ? '▸' : '▾'}</span>
          <span className="sidebar-label">THREADS</span>
        </button>
        <button className="icon-btn" onClick={onCreate} title="New thread">+</button>
      </div>

      {!threadsCollapsed && (
        <ul className="thread-list">
          {threads.map(t => (
            <li
              key={t.id}
              className={`thread-item${t.id === activeId ? ' active' : ''}`}
              onClick={() => onSelect(t.id)}
            >
              {editing === t.id ? (
                <input
                  className="thread-rename-input"
                  value={editVal}
                  autoFocus
                  onChange={e => setEditVal(e.target.value)}
                  onBlur={() => commitEdit(t.id)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') commitEdit(t.id)
                    if (e.key === 'Escape') setEditing(null)
                  }}
                  onClick={e => e.stopPropagation()}
                />
              ) : (
                <>
                  <span className="thread-name" onDoubleClick={e => startEdit(e, t)}>
                    {t.name}
                  </span>
                  <span className="thread-count">
                    {t.messages.length > 0 ? Math.ceil(t.messages.length / 2) : ''}
                  </span>
                  {threads.length > 1 && (
                    <button
                      className="thread-delete"
                      onClick={e => { e.stopPropagation(); onDelete(t.id) }}
                      title="Delete thread"
                    >×</button>
                  )}
                </>
              )}
            </li>
          ))}
        </ul>
      )}

      <section className="tools-panel">
        <button
          className="tools-header"
          onClick={() => setToolsCollapsed(v => !v)}
          title={toolsCollapsed ? 'Expand tools panel' : 'Collapse tools panel'}
        >
          <span className="menu-label">
            <span className="menu-caret">{toolsCollapsed ? '▸' : '▾'}</span>
            <span className="sidebar-label">TOOLS</span>
          </span>
          <span className="tools-total-badge">{totalTools}</span>
        </button>

        {!toolsCollapsed && (
          <div className="tools-body">
            {providerEntries.length === 0 ? (
              <div className="tools-empty">No tools connected</div>
            ) : (
              providerEntries.map(([provider, tools]) => {
                const list = Array.isArray(tools) ? tools : []
                const isOpen = !!openProviders[provider]
                return (
                  <div className="provider-block" key={provider}>
                    <button className="provider-header" onClick={() => toggleProvider(provider)}>
                      <span>{provider}</span>
                      <span className="provider-count">{list.length}</span>
                    </button>
                    {isOpen && (
                      <ul className="tool-list">
                        {list.map((tool, idx) => {
                          const name = typeof tool === 'string' ? tool : tool?.name
                          if (!name) return null
                          const shortName =
                            typeof tool === 'object' && tool?.short_name
                              ? tool.short_name
                              : name.includes('_') ? name.split('_').slice(1).join('_') : name
                          const description =
                            typeof tool === 'object' && tool?.description
                              ? tool.description
                              : ''
                          const isActive = activeToolSet.has(name)
                          return (
                            <li
                              key={`${provider}-${name}-${idx}`}
                              className={`tool-item${isActive ? ' active' : ''}`}
                              title={description || name}
                            >
                              {shortName}
                            </li>
                          )
                        })}
                      </ul>
                    )}
                  </div>
                )
              })
            )}
          </div>
        )}
      </section>

      <section className="rag-panel">
        <button
          className="rag-header"
          onClick={() => setContextCollapsed(v => !v)}
          title={contextCollapsed ? 'Expand context panel' : 'Collapse context panel'}
        >
          <span className="menu-label">
            <span className="menu-caret">{contextCollapsed ? '▸' : '▾'}</span>
            <span className="sidebar-label">CONTEXT</span>
          </span>
          <div className="menu-pill-row">
            <span className="rag-status-pill rag-status-pill--on">{effectiveN} MSGs</span>
          </div>
        </button>
        {!contextCollapsed && (
          <div className="rag-body context-body">
            <div className="context-row">
              <span className="context-k">RAG injection</span>
              <span className={`context-v${ragModeClass}`}>{ragModeLabel}</span>
            </div>
            <div className="context-row">
              <span className="context-k">temperature</span>
              <span className="context-v on">{tempLabel}</span>
            </div>
            <div className="context-row">
              <span className="context-k">context</span>
              <span className="context-v on">
                {estimatedContextTokens} tok{ctxLength ? ` / ${ctxLength}` : ' est'}
              </span>
            </div>
            <div className="sidebar-runtime-meter" title="Approximate pre-send context token usage">
              <div className="sidebar-runtime-meter-track">
                <div
                  className="sidebar-runtime-meter-fill"
                  style={{ width: `${contextPct}%` }}
                />
              </div>
            </div>
            <div className="context-block">
              <div className="context-label">SYSTEM PROMPT</div>
              <div className="context-text">{contextSystemPrompt || '(none)'}</div>
            </div>
            <div className="context-block">
              <div className="context-label">PRE-SEND WINDOW ({contextPreviewMessages.length})</div>
              <div className="context-list">
                {contextPreviewMessages.length === 0 && (
                  <div className="rag-hint">No messages in context yet.</div>
                )}
                {contextPreviewMessages.map((m, idx) => (
                  <div className={`context-item context-item--${m.role}`} key={`${m.role}-${idx}`}>
                    <span className="context-role">{m.role}</span>
                    <span className="context-msg">{contentPreview(m.content)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="rag-panel">
        <div className="rag-toggle-row">
          <span className="menu-label">
            <span className={`dot ${ragEnabled ? 'dot--green' : 'dot--gray'}`} />
            <span className="sidebar-label">ENABLE RAG</span>
          </span>
          <button
            type="button"
            className={`rag-toggle-btn${ragEnabled ? ' active' : ''}${!ragUsable ? ' disabled' : ''}`}
            onClick={() => {
              if (!ragUsable) return
              onToggleRagEnabled(!ragEnabled)
            }}
            disabled={!ragUsable}
            title={
              ragUsable
                ? (ragEnabled ? 'Disable RAG context injection' : 'Enable RAG context injection')
                : 'RAG toggle unavailable until RAG is connected'
            }
          >
            {ragEnabled ? 'ON' : 'OFF'}
          </button>
        </div>
      </section>

      <div className="sidebar-footer">
        <div className="footer-main-row">
          <div className="footer-badges">
            <span className={`storage-badge storage-badge--${useDb ? 'db' : 'local'}`}>
              {useDb ? 'DB' : 'LOCAL'}
            </span>
            {Number(ragVectors) > 0 && (
              <span className="storage-badge storage-badge--rag">
                {Math.max(0, Number(ragVectors) || 0)} VEC
              </span>
            )}
          </div>
          <button className="icon-btn" onClick={onGear} title="Settings">⚙</button>
        </div>
        <div className="footer-legal">
          <div className="footer-legal-brand">
            <img className="footer-legal-logo" src="/miniloader-icon.ico" alt="" />
            <a className="footer-legal-link" href="https://miniloader.ai" target="_blank" rel="noreferrer">
              Copyright 2026 Miniloader.ai
            </a>
          </div>
          <div className="footer-legal-links">
            <a className="footer-legal-link" href="https://miniloader.ai/tos" target="_blank" rel="noreferrer">Terms</a>
            <span className="footer-legal-sep">|</span>
            <a className="footer-legal-link" href="https://miniloader.ai/privacy" target="_blank" rel="noreferrer">Privacy</a>
          </div>
        </div>
      </div>
    </aside>
  )
}

// ── Message ───────────────────────────────────────────────────────────────────

function renderWithCitations(text) {
  const parts = String(text || '').split(/(\[source:[^\]]+\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[source:([^\]]+)\]$/)
    if (m) {
      const source = m[1]
      return (
        <span key={i} className="citation-badge" title={source}>
          {source}
        </span>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function ToolCallBadge({ tc }) {
  const shortName = tc.tool_name?.includes('_')
    ? tc.tool_name.split('_').slice(1).join('_')
    : tc.tool_name
  const isPending = tc.status === 'pending'
  return (
    <span className={`tool-call-badge${isPending ? ' pending' : ' completed'}`}>
      <span className="tool-call-icon">{isPending ? '⟳' : '✓'}</span>
      <span className="tool-call-name">{shortName}</span>
    </span>
  )
}

function Message({ msg, showStreamCursor, toolCalls }) {
  const isUser = msg.role === 'user'
  const parts = Array.isArray(msg.content) ? msg.content : null
  const cursor = showStreamCursor ? (
    <span className="cursor" aria-hidden="true">▋</span>
  ) : null
  const hasToolCalls = !isUser && Array.isArray(toolCalls) && toolCalls.length > 0

  return (
    <div className={`msg msg--${isUser ? 'user' : 'assistant'}`}>
      <div className="msg-role">{isUser ? 'you' : 'assistant'}</div>
      <div className="msg-bubble">
        {hasToolCalls && (
          <div className="tool-calls-row">
            {toolCalls.map(tc => (
              <ToolCallBadge key={tc.tool_call_id} tc={tc} />
            ))}
          </div>
        )}
        {parts ? (
          <div className="msg-multipart">
            {parts.map((part, i) => {
              if (part?.type === 'text') {
                return (
                  <span key={i} className="msg-text msg-text--block">{part.text ?? ''}</span>
                )
              }
              if (part?.type === 'image_url' && part.image_url?.url) {
                return (
                  <img
                    key={i}
                    className="msg-image"
                    src={part.image_url.url}
                    alt=""
                  />
                )
              }
              return null
            })}
            {cursor}
          </div>
        ) : (
          <span className="msg-inline">
            <span className="msg-text">
              {isUser ? msg.content : renderWithCitations(msg.content)}
            </span>
            {cursor}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Status dot ───────────────────────────────────────────────────────────────

function Dot({ color, label }) {
  return (
    <span className="status-chip">
      <span className={`dot dot--${color}`} />
      {label}
    </span>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const saved       = loadLocal()
  const initThreads = saved?.threads?.length > 0 ? saved.threads : [makeThread(1)]
  const initActive  = saved?.activeId || initThreads[0].id

  const [threads,      setThreads]      = useState(initThreads)
  const [activeId,     setActiveId]     = useState(initActive)
  const [input,        setInput]        = useState('')
  const [streaming,    setStreaming]    = useState(false)
  const [llmReady,        setLlmReady]        = useState(false)
  const [agentConnected,  setAgentConnected]  = useState(false)
  const [ragConnected,    setRagConnected]    = useState(false)
  const [ragVectors,   setRagVectors]   = useState(0)
  const [ragMetrics, setRagMetrics] = useState({
    total_queries: 0,
    avg_top1_score: 0,
    avg_latency_ms: 0,
    empty_result_rate: 0,
  })
  const [ctxLength,    setCtxLength]    = useState(null)
  const [wsStatus,     setWsStatus]     = useState('connecting')
  const [tunnelUrl,    setTunnelUrl]    = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState(loadPrompt)
  const [maxContextMessages, setMaxContextMessages] = useState(loadMaxContextMessages)
  const [toolProviders, setToolProviders] = useState({})
  const [activeToolCalls, setActiveToolCalls] = useState({})
  const [dbStatus,     setDbStatus]     = useState({
    connected: false, db_exists: false, tables_ready: false, tables: [], db_path: null,
  })
  const [moeEnabled, setMoeEnabled] = useState(loadMoeEnabled)
  const [ragEnabled, setRagEnabled] = useState(loadRagEnabled)
  const [ragContextMode, setRagContextMode] = useState('disabled')
  const [pendingImages, setPendingImages] = useState([])
  const [streamCursorMessageId, setStreamCursorMessageId] = useState(null)
  const [lastSentPayload, setLastSentPayload] = useState(null)
  const [rawJsonOpen, setRawJsonOpen] = useState(false)
  const [demoDismissed, setDemoDismissed] = useState(false)
  const [lastUsedTemperature, setLastUsedTemperature] = useState(0.7)
  const [streamToolCalls, setStreamToolCalls] = useState([])
  const [voiceAvailable, setVoiceAvailable] = useState(false)
  const [voiceActive, setVoiceActive] = useState(false)
  const [voiceBusy, setVoiceBusy] = useState(false)
  const [isNarrowViewport, setIsNarrowViewport] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  const bottomRef   = useRef(null)
  const inputRef    = useRef(null)
  const fileRef     = useRef(null)
  const wsRef       = useRef(null)
  const abortRef    = useRef(null)
  const streamingId = useRef(null)
  const activeRequestIdRef = useRef(null)
  const loadedRef   = useRef(new Set())
  const dbSettingsReadyRef = useRef(false)
  const livekitRoomRef = useRef(null)
  const voiceMessagesRef = useRef({})
  const voiceTurnToolCallsRef = useRef({})
  const useDbRef = useRef(false)
  const ensureVoiceThreadRef = useRef(null)

  const useDb  = dbStatus.connected && dbStatus.tables_ready
  const active = threads.find(t => t.id === activeId) ?? threads[0]

  const ensureVoiceThread = useCallback(async () => {
    const existing = threads.find(t => t.id === VOICE_THREAD_ID)
    if (existing) {
      setActiveId(existing.id)
      return existing.id
    }
    const voiceThread = { id: VOICE_THREAD_ID, name: VOICE_THREAD_NAME, messages: [] }
    if (useDb) await dbApi.createThread(voiceThread.id, voiceThread.name)
    setThreads(prev => {
      if (prev.some(t => t.id === VOICE_THREAD_ID)) return prev
      return [...prev, voiceThread]
    })
    setActiveId(VOICE_THREAD_ID)
    loadedRef.current.add(VOICE_THREAD_ID)
    return VOICE_THREAD_ID
  }, [threads, useDb])

  const disconnectVoiceRoom = useCallback(() => {
    const room = livekitRoomRef.current
    if (room) {
      try { room.disconnect() } catch { /* ignore */ }
      livekitRoomRef.current = null
    }
    setVoiceActive(false)
  }, [])

  const connectVoiceRoom = useCallback(async () => {
    if (voiceBusy || voiceActive) return
    setVoiceBusy(true)
    try {
      await ensureVoiceThread()
      const tokenRes = await authedFetch('/voice/token')
      if (!tokenRes.ok) {
        const detail = await tokenRes.text()
        throw new Error(detail || `voice token failed (${tokenRes.status})`)
      }
      const tokenJson = await tokenRes.json()
      const room = new Room({ adaptiveStream: true, dynacast: true })
      room.on(RoomEvent.Disconnected, () => setVoiceActive(false))
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach()
          el.autoplay = true
          el.setAttribute('playsinline', '')
          document.body.appendChild(el)
          el.play().catch(() => {
            // Retry on next user gesture — browsers may block autoplay
            const resume = () => { el.play().catch(() => {}); document.removeEventListener('click', resume) }
            document.addEventListener('click', resume, { once: true })
          })
        }
      })
      await room.connect(tokenJson.url, tokenJson.token)
      await room.localParticipant.setMicrophoneEnabled(true)
      livekitRoomRef.current = room
      setVoiceActive(true)
    } finally {
      setVoiceBusy(false)
    }
  }, [voiceBusy, voiceActive, ensureVoiceThread])

  useEffect(() => {
    useDbRef.current = useDb
  }, [useDb])

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 768px)')
    const apply = () => setIsNarrowViewport(mq.matches)
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  useEffect(() => {
    ensureVoiceThreadRef.current = ensureVoiceThread
  }, [ensureVoiceThread])

  useEffect(() => {
    savePrompt(systemPrompt)
    if (useDb && dbSettingsReadyRef.current) {
      dbApi.setSetting('system_prompt', systemPrompt)
    }
  }, [systemPrompt, useDb])

  useEffect(() => {
    saveMaxContextMessages(maxContextMessages)
    if (useDb && dbSettingsReadyRef.current) {
      dbApi.setSetting(
        'max_context_messages',
        maxContextMessages == null ? '' : String(maxContextMessages),
      )
    }
  }, [maxContextMessages, useDb])

  useEffect(() => {
    saveMoeEnabled(moeEnabled)
    if (useDb && dbSettingsReadyRef.current) {
      dbApi.setSetting('moe_enabled', moeEnabled ? '1' : '0')
    }
  }, [moeEnabled, useDb])

  useEffect(() => {
    saveRagEnabled(ragEnabled)
    if (useDb && dbSettingsReadyRef.current) {
      dbApi.setSetting('rag_enabled', ragEnabled ? '1' : '0')
    }
  }, [ragEnabled, useDb])

  // ── Effective n: user override or derived from ctx_length ─────────────────
  const effectiveN = maxContextMessages ?? (ctxLength
    ? Math.max(2, Math.min(50, Math.floor((ctxLength * 0.8) / 150)))
    : 20)

  // ── Persist to localStorage (when in local mode) ────────────────────────
  useEffect(() => {
    if (!useDb) saveLocal(threads, activeId)
  }, [threads, activeId, useDb])

  // ── Poll DB status ───────────────────────────────────────────────────────
  useEffect(() => {
    let alive = true
    const poll = async () => {
      const s = await dbApi.status()
      if (alive && !s.error) setDbStatus(s)
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const refreshRagMetrics = useCallback(async () => {
    const metrics = await ragApi.metrics()
    if (!metrics?.error) setRagMetrics(metrics)
  }, [])

  useEffect(() => {
    if (!showSettings) return
    let alive = true
    const tick = async () => {
      const metrics = await ragApi.metrics()
      if (alive && !metrics?.error) setRagMetrics(metrics)
    }
    tick()
    const timer = setInterval(tick, 30000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [showSettings])

  // ── Load threads from DB when DB becomes ready ──────────────────────────
  useEffect(() => {
    if (!useDb) return
    let alive = true
    ;(async () => {
      const rows = await dbApi.threads()
      if (!alive || rows.error) return
      if (rows.length === 0) {
        const t = makeThread(1)
        await dbApi.createThread(t.id, t.name)
        if (alive) { setThreads([t]); setActiveId(t.id) }
      } else {
        const loaded = rows.map(r => ({ id: r.id, name: r.name, messages: [] }))
        if (alive) {
          setThreads(loaded)
          loadedRef.current = new Set()
          const firstId = loaded[0].id
          setActiveId(firstId)
        }
      }
    })()
    return () => { alive = false }
  }, [useDb])

  useEffect(() => {
    if (!useDb) {
      dbSettingsReadyRef.current = false
      return
    }
    let alive = true
    dbSettingsReadyRef.current = false
    ;(async () => {
      const settings = await dbApi.getSettings()
      if (!alive || settings?.error || typeof settings !== 'object' || settings == null) {
        dbSettingsReadyRef.current = true
        return
      }
      if (Object.prototype.hasOwnProperty.call(settings, 'system_prompt')) {
        const next = String(settings.system_prompt || '')
        setSystemPrompt(next)
        savePrompt(next)
      }
      if (Object.prototype.hasOwnProperty.call(settings, 'max_context_messages')) {
        const raw = String(settings.max_context_messages ?? '').trim()
        if (!raw) {
          setMaxContextMessages(null)
          saveMaxContextMessages(null)
        } else {
          const n = parseInt(raw, 10)
          const v = Number.isFinite(n) && n >= 2 ? n : null
          setMaxContextMessages(v)
          saveMaxContextMessages(v)
        }
      }
      if (Object.prototype.hasOwnProperty.call(settings, 'moe_enabled')) {
        const next = parseSettingBool(settings.moe_enabled, false)
        setMoeEnabled(next)
        saveMoeEnabled(next)
      }
      if (Object.prototype.hasOwnProperty.call(settings, 'rag_enabled')) {
        const next = parseSettingBool(settings.rag_enabled, true)
        setRagEnabled(next)
        saveRagEnabled(next)
      }
      dbSettingsReadyRef.current = true
    })()
    return () => { alive = false }
  }, [useDb])

  // ── Lazy-load messages when a thread is selected (DB mode) ──────────────
  useEffect(() => {
    if (!useDb || !activeId) return
    if (loadedRef.current.has(activeId)) return
    let alive = true
    ;(async () => {
      const rows = await dbApi.messages(activeId)
      if (!alive || rows.error) return
      loadedRef.current.add(activeId)
      const msgs = rows.map(r => ({
        id: r.id, role: r.role, content: parseStoredContent(r.content),
      }))
      setThreads(prev => prev.map(t =>
        t.id === activeId ? { ...t, messages: msgs } : t
      ))
    })()
    return () => { alive = false }
  }, [activeId, useDb])

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active?.messages])

  // ── WebSocket ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws

      ws.onopen  = () => setWsStatus('connected')
      ws.onclose = () => {
        setWsStatus('disconnected')
        setAgentConnected(false)
        setRagConnected(false)
        setRagVectors(0)
        setRagContextMode('disabled')
        setVoiceAvailable(false)
        disconnectVoiceRoom()
        if (!cancelled) setTimeout(connect, 3000)
      }
      ws.onerror = () => setWsStatus('disconnected')

      ws.onmessage = ({ data }) => {
        try {
          const msg = JSON.parse(data)
          if (msg.type === 'config') {
            setLlmReady(!!msg.llm_ready)
            setAgentConnected(!!msg.agent_connected)
            if (msg.ctx_length != null) setCtxLength(msg.ctx_length)
            setRagConnected(!!msg.rag_connected)
            setRagVectors(Number.isFinite(msg.rag_vector_count) ? msg.rag_vector_count : 0)
            setVoiceAvailable(!!msg.voice_available)
            if (!msg.voice_available) disconnectVoiceRoom()
          }
          if (msg.type === 'tunnel_url') setTunnelUrl(msg.url || null)
          if (msg.type === 'tool_registry') {
            const providers = (msg.providers && typeof msg.providers === 'object') ? msg.providers : {}
            setToolProviders(providers)
            if (Array.isArray(msg.active_tool_calls)) {
              const next = {}
              for (const call of msg.active_tool_calls) {
                const id = call?.tool_call_id
                const toolName = call?.tool_name
                if (id && toolName) next[id] = { tool_name: toolName }
              }
              setActiveToolCalls(next)
            }
          }
          if (msg.type === 'tool_call') {
            const id = msg.tool_call_id
            const toolName = msg.tool_name
            if (msg.status === 'pending' && id && toolName) {
              setActiveToolCalls(prev => ({ ...prev, [id]: { tool_name: toolName } }))
            } else if ((msg.status === 'result' || msg.status === 'completed') && id) {
              setActiveToolCalls(prev => {
                const next = { ...prev }
                delete next[id]
                return next
              })
            }
            if (msg.request_id && msg.request_id === activeRequestIdRef.current) {
              setStreamToolCalls(prev => {
                const entry = {
                  tool_call_id: id,
                  tool_name: toolName,
                  status: msg.status,
                  args: msg.args,
                  result: msg.result,
                }
                const idx = prev.findIndex(tc => tc.tool_call_id === id)
                if (idx >= 0) {
                  const copy = [...prev]
                  copy[idx] = entry
                  return copy
                }
                return [...prev, entry]
              })
            }
            if (msg.request_id && voiceMessagesRef.current[msg.request_id]) {
              const requestId = String(msg.request_id)
              const entry = voiceMessagesRef.current[requestId]
              const calls = voiceTurnToolCallsRef.current
              if (!calls[requestId]) calls[requestId] = []
              const toolEntry = {
                tool_call_id: id,
                tool_name: toolName,
                status: msg.status,
                args: msg.args,
                result: msg.result,
              }
              const idx = calls[requestId].findIndex(tc => tc.tool_call_id === id)
              if (idx >= 0) {
                calls[requestId][idx] = { ...calls[requestId][idx], ...toolEntry }
              } else {
                calls[requestId].push(toolEntry)
              }
              setThreads(prev => prev.map(t => (
                t.id !== entry.threadId ? t : {
                  ...t,
                  messages: t.messages.map(m => (
                    m.id === entry.asstId ? { ...m, toolCalls: [...calls[requestId]] } : m
                  )),
                }
              )))
            }
          }
          if (msg.type === 'tool_result' && msg.tool_call_id) {
            const id = msg.tool_call_id
            setActiveToolCalls(prev => {
              if (!prev[id]) return prev
              const next = { ...prev }
              delete next[id]
              return next
            })
          }
          if (msg.type === 'rag_debug') {
            if (msg.mode === 'auto' || msg.mode === 'tool' || msg.mode === 'disabled') {
              setRagContextMode(msg.mode)
            }
          }
          if (msg.type === 'voice_turn') {
            const requestId = String(msg.request_id || '')
            const threadId = String(msg.thread_id || VOICE_THREAD_ID)
            if (msg.phase === 'start') {
              if (voiceMessagesRef.current[requestId]) return
              const userText = String(msg.user_text || '').trim()
              const userId = userText ? `voice_user_${requestId}` : null
              const asstId = `voice_asst_${requestId}`
              const nextMap = { ...voiceMessagesRef.current }
              nextMap[requestId] = { userId, asstId, threadId }
              voiceMessagesRef.current = nextMap
              ensureVoiceThreadRef.current?.().catch(() => {})
              setThreads(prev => prev.map(t => {
                if (t.id !== threadId) return t
                const msgs = [...t.messages]
                if (userText && userId) {
                  msgs.push({ id: userId, role: 'user', content: userText })
                }
                msgs.push({ id: asstId, role: 'assistant', content: '', streaming: true })
                return { ...t, messages: msgs }
              }))
              setStreamCursorMessageId(asstId)
              if (useDbRef.current) {
                if (userText && userId) {
                  dbApi.createMessage(threadId, { id: userId, role: 'user', content: userText })
                }
              }
            } else if (msg.phase === 'token') {
              const entry = voiceMessagesRef.current[requestId]
              if (!entry?.asstId) return
              const delta = String(msg.content || '')
              if (!delta) return
              setThreads(prev => prev.map(t => (
                t.id !== entry.threadId ? t : {
                  ...t,
                  messages: t.messages.map(m => (
                    m.id === entry.asstId ? { ...m, content: String(m.content || '') + delta } : m
                  )),
                }
              )))
            } else if (msg.phase === 'end' || msg.phase === 'error') {
              const entry = voiceMessagesRef.current[requestId]
              if (!entry?.asstId) return
              const finalText = msg.phase === 'error'
                ? `Error: ${String(msg.error || 'voice turn failed')}`
                : String(msg.assistant_text || '')
              setThreads(prev => prev.map(t => (
                t.id !== entry.threadId ? t : {
                  ...t,
                  messages: t.messages.map(m => (
                    m.id === entry.asstId ? { ...m, content: finalText, streaming: false } : m
                  )),
                }
              )))
              if (useDbRef.current && finalText) {
                dbApi.createMessage(entry.threadId, { id: entry.asstId, role: 'assistant', content: finalText })
              }
              setStreamCursorMessageId(prev => (prev === entry.asstId ? null : prev))
              const nextMap = { ...voiceMessagesRef.current }
              delete nextMap[requestId]
              voiceMessagesRef.current = nextMap
              const nextCalls = { ...voiceTurnToolCallsRef.current }
              delete nextCalls[requestId]
              voiceTurnToolCallsRef.current = nextCalls
            }
          }
        } catch { /* ignore */ }
      }
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
    }
  }, [])

  useEffect(() => {
    return () => disconnectVoiceRoom()
  }, [disconnectVoiceRoom])

  // ── Thread helpers ────────────────────────────────────────────────────────
  const updateThread = useCallback((id, fn) => {
    setThreads(prev => prev.map(t => t.id === id ? fn(t) : t))
  }, [])

  const createThread = async () => {
    const t = makeThread(threads.length + 1)
    if (useDb) await dbApi.createThread(t.id, t.name)
    setThreads(prev => [...prev, t])
    setActiveId(t.id)
    loadedRef.current.add(t.id)
    setMobileNavOpen(false)
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  const deleteThread = async (id) => {
    if (useDb) await dbApi.deleteThread(id)
    const rest = threads.filter(t => t.id !== id)
    loadedRef.current.delete(id)
    if (rest.length === 0) {
      const t = makeThread(1)
      if (useDb) await dbApi.createThread(t.id, t.name)
      setThreads([t])
      setActiveId(t.id)
      loadedRef.current.add(t.id)
    } else {
      setThreads(rest)
      if (activeId === id) setActiveId(rest[0]?.id)
    }
  }

  const renameThread = async (id, name) => {
    if (useDb) await dbApi.renameThread(id, name)
    setThreads(prev => prev.map(t => t.id === id ? { ...t, name } : t))
  }

  const selectThread = (id) => {
    setActiveId(id)
    if (showSettings) setShowSettings(false)
    setMobileNavOpen(false)
  }

  // ── Init DB from settings panel ──────────────────────────────────────────
  const handleInitDb = async () => {
    const r = await dbApi.init()
    if (!r.error) {
      const s = await dbApi.status()
      if (!s.error) setDbStatus(s)
    }
  }

  const onPickImages = (e) => {
    const files = Array.from(e.target.files || [])
    e.target.value = ''
    if (!files.length) return
    setPendingImages(prev => {
      const room = MAX_ATTACH_IMAGES - prev.length
      if (room <= 0) return prev
      const take = files.slice(0, room)
      const next = [...prev]
      for (const file of take) {
        if (!file.type.startsWith('image/')) continue
        if (file.size > MAX_IMAGE_BYTES) continue
        const id = genId()
        const reader = new FileReader()
        reader.onload = () => {
          const url = typeof reader.result === 'string' ? reader.result : ''
          if (!url) return
          setPendingImages(cur => {
            if (cur.some(x => x.id === id)) {
              return cur.map(x => (x.id === id ? { ...x, dataUrl: url } : x))
            }
            return cur
          })
        }
        next.push({ id, dataUrl: '' })
        reader.readAsDataURL(file)
      }
      return next
    })
  }

  const removePendingImage = (id) => {
    setPendingImages(prev => prev.filter(p => p.id !== id))
  }

  // ── Send ──────────────────────────────────────────────────────────────────
  const sendMessage = async () => {
    const text = input.trim()
    const readyImages = pendingImages.filter(p => p.dataUrl)
    if (streaming) return
    if (!text && readyImages.length === 0) return

    const threadId = active.id
    const requestId = `web_${genId()}`
    const userContent = (() => {
      const imageParts = readyImages.map(p => ({
        type: 'image_url',
        image_url: { url: p.dataUrl },
      }))
      if (!text && imageParts.length) return imageParts
      if (!imageParts.length) return text
      return [{ type: 'text', text }, ...imageParts]
    })()

    const userMsg  = { id: genId(), role: 'user',      content: userContent }
    const asstMsg  = { id: genId(), role: 'assistant', content: '', streaming: true }

    streamingId.current = asstMsg.id
    activeRequestIdRef.current = requestId
    setStreamCursorMessageId(asstMsg.id)
    setStreamToolCalls([])

    updateThread(threadId, t => ({
      ...t,
      messages: [...t.messages, userMsg, asstMsg],
    }))
    setInput('')
    setPendingImages([])
    setStreaming(true)

    if (useDb) {
      dbApi.createMessage(threadId, {
        id: userMsg.id,
        role: userMsg.role,
        content: contentForDb(userMsg.content),
      })
    }

    const allMsgs = [...active.messages, userMsg].map(({ role, content }) => ({ role, content }))
    const history = allMsgs.slice(-effectiveN)
    if (systemPrompt.trim()) {
      history.unshift({ role: 'system', content: systemPrompt.trim() })
    }

    abortRef.current = new AbortController()
    let fullContent = ''
    let sawAnyDelta = false
    console.info(`[gpt_terminal:web] send request_id=${requestId} thread_id=${threadId}`)
    try {
      const chatBody = {
        request_id: requestId,
        thread_id: threadId,
        messages: history,
        rag_enabled: ragEnabled,
        temperature: 0.7,
      }
      if (moeEnabled) chatBody.moe_enabled = true
      setLastSentPayload(chatBody)
      setLastUsedTemperature(Number(chatBody.temperature ?? 0.7))

      const res = await authedFetch('/api/chat', {
        method:  'POST',
        headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
        body:    JSON.stringify(chatBody),
        signal:  abortRef.current.signal,
      })
      console.info(`[gpt_terminal:web] /api/chat status=${res.status} request_id=${requestId}`)

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })

        const lines = buf.split('\n')
        buf = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6).trim()
          if (payload === '[DONE]') continue
          try {
            const parsed = JSON.parse(payload)
            const delta  = parsed.choices?.[0]?.delta?.content ?? ''
            if (delta) {
              sawAnyDelta = true
              fullContent += delta
              updateThread(threadId, t => ({
                ...t,
                messages: t.messages.map(m =>
                  m.id === asstMsg.id ? { ...m, content: m.content + delta } : m
                ),
              }))
            }
            if (parsed.error) throw new Error(parsed.error)
          } catch (parseErr) {
            if (parseErr.message && !parseErr.message.includes('JSON')) throw parseErr
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      fullContent = fullContent || `Error: ${err.message}`
      console.error(`[gpt_terminal:web] stream error request_id=${requestId}:`, err)
      updateThread(active.id, t => ({
        ...t,
        messages: t.messages.map(m =>
          m.id === streamingId.current
            ? { ...m, content: m.content || `Error: ${err.message}` }
            : m
        ),
      }))
    } finally {
      const finalizedContent = typeof fullContent === 'string'
        ? fullContent.replace(/\s+$/u, '')
        : fullContent
      console.info(
        `[gpt_terminal:web] stream done request_id=${requestId} ` +
        `tokens_received=${sawAnyDelta ? 'yes' : 'no'} chars=${String(finalizedContent || '').length}`
      )
      updateThread(active.id, t => ({
        ...t,
        messages: t.messages.map(m =>
          m.id === streamingId.current
            ? {
                ...m,
                content: typeof m.content === 'string' ? m.content.replace(/\s+$/u, '') : m.content,
                streaming: false,
              }
            : m
        ),
      }))
      if (useDb && finalizedContent) {
        dbApi.createMessage(threadId, { id: asstMsg.id, role: 'assistant', content: finalizedContent })
      }
      streamingId.current = null
      activeRequestIdRef.current = null
      setStreamCursorMessageId(null)
      setStreaming(false)
    }
  }

  const stopStreaming = () => {
    setStreamCursorMessageId(null)
    activeRequestIdRef.current = null
    abortRef.current?.abort()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  // ── Derived status ────────────────────────────────────────────────────────
  const wsColor    = wsStatus === 'connected'  ? 'green'
                   : wsStatus === 'connecting' ? 'yellow'
                   :                             'red'
  const agentColor = agentConnected ? 'green' : 'gray'
  const llmColor   = llmReady ? 'green' : 'red'
  const dbColor    = useDb ? 'green' : dbStatus.connected ? 'yellow' : 'red'
  // Treat RAG as usable once the engine is wired/running, even before vectors
  // are ingested. An empty index is still a valid "enabled" state.
  const ragColor = ragConnected ? 'green' : 'red'
  const ragUsable = ragConnected
  const activePersona = derivePersonaId(systemPrompt)
  const contextPreviewMessages = (() => {
    const base = Array.isArray(active?.messages) ? active.messages : []
    const draftText = input.trim()
    const draftImages = pendingImages.filter(p => p.dataUrl)
    const hasDraft = draftText || draftImages.length > 0
    const draftContent = (() => {
      if (!hasDraft) return null
      const imageParts = draftImages.map(p => ({ type: 'image_url', image_url: { url: p.dataUrl } }))
      if (!draftText && imageParts.length) return imageParts
      if (!imageParts.length) return draftText
      return [...imageParts, { type: 'text', text: draftText }]
    })()
    const withDraft = hasDraft ? [...base, { role: 'user', content: draftContent }] : base
    const sliced = withDraft.slice(-effectiveN)
    return sliced.map(({ role, content }) => ({ role, content }))
  })()

  const handlePersonaSelect = useCallback((personaId) => {
    const preset = PERSONA_PRESETS.find(p => p.id === personaId)
    if (preset) setSystemPrompt(preset.prompt)
  }, [])

  useEffect(() => {
    if (!ragUsable && ragEnabled) {
      setRagEnabled(false)
    }
  }, [ragUsable, ragEnabled])
  useEffect(() => {
    if (!ragEnabled) {
      setRagContextMode('disabled')
    }
  }, [ragEnabled])

  const shellClass = [
    'shell',
    isNarrowViewport ? 'shell--narrow' : '',
    isNarrowViewport && mobileNavOpen && !showSettings ? 'shell--nav-open' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className={shellClass}>
      {isNarrowViewport && mobileNavOpen && !showSettings && (
        <button
          type="button"
          className="mobile-nav-backdrop"
          aria-label="Close menu"
          onClick={() => setMobileNavOpen(false)}
        />
      )}
      {showSettings ? (
        <SettingsPanel
          dbStatus={dbStatus}
          onInitDb={handleInitDb}
          systemPrompt={systemPrompt}
          onPromptChange={setSystemPrompt}
          maxContextMessages={maxContextMessages}
          onMaxContextMessagesChange={setMaxContextMessages}
          ctxLength={ctxLength}
          ragMetrics={ragMetrics}
          onRefreshRagMetrics={refreshRagMetrics}
          activePersona={activePersona}
          onPersonaSelect={handlePersonaSelect}
          onOpenRawJson={() => setRawJsonOpen(true)}
          onCloseRawJson={() => setRawJsonOpen(false)}
          rawJsonOpen={rawJsonOpen}
          rawJsonText={lastSentPayload ? JSON.stringify(lastSentPayload, null, 2) : '{\n  "info": "No chat request has been sent in this session yet."\n}'}
          onClose={() => {
            setShowSettings(false)
            setMobileNavOpen(false)
          }}
        />
      ) : (
        <Sidebar
          threads={threads}
          activeId={activeId}
          onSelect={selectThread}
          onCreate={createThread}
          onDelete={deleteThread}
          onRename={renameThread}
          dbStatus={dbStatus}
          onGear={() => {
            setShowSettings(true)
            setMobileNavOpen(false)
          }}
          onRequestCloseMobile={() => setMobileNavOpen(false)}
          showMobileClose={isNarrowViewport}
          toolProviders={toolProviders}
          activeToolCalls={activeToolCalls}
          ragConnected={ragConnected}
          ragVectors={ragVectors}
          ragEnabled={ragEnabled}
          ragUsable={ragUsable}
          ragContextMode={ragContextMode}
          onToggleRagEnabled={setRagEnabled}
          effectiveN={effectiveN}
          contextSystemPrompt={systemPrompt.trim()}
          contextPreviewMessages={contextPreviewMessages}
          temperatureUsed={lastUsedTemperature}
          ctxLength={ctxLength}
        />
      )}

      <div className="chat-panel">
        {!demoDismissed && (
          <div className="demo-notice" role="note">
            <span>{agentConnected
              ? 'Agent mode active. Tool calls are handled automatically by the agent engine.'
              : 'Demo only: Agent Smith is limited right now. It cannot browse the web and only uses document context via RAG.'
            }</span>
            <button className="demo-notice-close" onClick={() => setDemoDismissed(true)} title="Dismiss">✕</button>
          </div>
        )}
        <header className="chat-header">
          <div className="header-left">
            {isNarrowViewport && !showSettings && (
              <button
                type="button"
                className="mobile-menu-btn"
                onClick={() => setMobileNavOpen(v => !v)}
                aria-expanded={mobileNavOpen}
                aria-controls="gpt-sidebar"
                title="Threads and tools"
              >
                ☰
              </button>
            )}
            <span className="header-title">{active?.name ?? 'Chat'}</span>
            {tunnelUrl && (
              <a className="tunnel-link" href={tunnelUrl} target="_blank" rel="noreferrer">
                {tunnelUrl}
              </a>
            )}
          </div>
          <div className="header-right">
            <div className="header-status">
              <Dot color={wsColor}    label="socket" />
              <Dot color={agentColor} label="AGENT" />
              <Dot color={llmColor}   label="LLM" />
              <Dot color={dbColor}    label="DB" />
              <Dot color={ragColor}   label="RAG" />
            </div>
          </div>
        </header>

        <div className="messages">
          {active?.messages.length === 0 && (
            <div className="empty">
              <div className="empty-icon">⚡</div>
              <p className="empty-heading">Start a conversation</p>
              <p className="empty-sub">
                {agentConnected
                  ? 'Agent engine is connected with tools ready.'
                  : llmReady
                    ? 'LLM is connected and ready.'
                    : 'Waiting for LLM server to connect\u2026'}
              </p>
            </div>
          )}
          {active?.messages.map(msg => (
            <Message
              key={msg.id}
              msg={msg}
              showStreamCursor={
                msg.role === 'assistant'
                && streamCursorMessageId === msg.id
                && msg.streaming
              }
              toolCalls={
                msg.role === 'assistant' && streamingId.current === msg.id
                  ? streamToolCalls
                  : undefined
              }
            />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="input-row">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="file-input-hidden"
            onChange={onPickImages}
          />
          <button
            type="button"
            className="attach-btn"
            onClick={() => fileRef.current?.click()}
            disabled={!llmReady || streaming || pendingImages.length >= MAX_ATTACH_IMAGES}
            title="Attach images"
          >
            ⧉
          </button>
          {voiceAvailable && (
            <button
              type="button"
              className={`voice-btn${voiceActive ? ' voice-btn--active' : ''}`}
              onClick={() => {
                if (voiceActive) disconnectVoiceRoom()
                else connectVoiceRoom().catch((err) => console.error('voice connect failed', err))
              }}
              disabled={voiceBusy}
              title={voiceActive ? 'Stop voice chat' : 'Start voice chat'}
            >
              {voiceBusy ? '...' : 'MIC'}
            </button>
          )}
          <div className="input-stack">
            {pendingImages.length > 0 && (
              <div className="pending-images">
                {pendingImages.map(p => (
                  <div key={p.id} className="pending-thumb-wrap">
                    {p.dataUrl ? (
                      <img className="pending-thumb" src={p.dataUrl} alt="" />
                    ) : (
                      <span className="pending-loading">…</span>
                    )}
                    <button
                      type="button"
                      className="pending-remove"
                      onClick={() => removePendingImage(p.id)}
                      title="Remove"
                    >×</button>
                  </div>
                ))}
              </div>
            )}
            <textarea
              ref={inputRef}
              className="input-box"
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder={llmReady ? 'Type a message\u2026 (Enter to send, Shift+Enter for newline)' : 'Waiting for LLM\u2026'}
              disabled={!llmReady && !streaming}
            />
          </div>
          {streaming
            ? <button className="send-btn send-btn--stop" onClick={stopStreaming} title="Stop">■</button>
            : (
              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={(!input.trim() && !pendingImages.some(p => p.dataUrl)) || !llmReady}
                title="Send"
              >↑</button>
            )}
        </div>
      </div>
    </div>
  )
}
