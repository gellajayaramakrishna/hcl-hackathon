// One-page chat UI: ask a question, see the answer + routing card, give 👍/👎, watch live stats.
import { useEffect, useRef, useState } from 'react'
import { askQuery, getSamples, getStats, sendFeedback } from './api.js'
import RoutingCard from './RoutingCard.jsx'
import TopBar from './TopBar.jsx'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [taskType, setTaskType] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [stats, setStats] = useState(null)
  const [samples, setSamples] = useState([])
  const endRef = useRef(null)

  const refreshStats = () => getStats().then(setStats).catch(e => setError(e.message))

  useEffect(() => {
    refreshStats()
    getSamples().then(setSamples).catch(() => {})
    const timer = setInterval(refreshStats, 10000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send() {
    const query = input.trim()
    if (!query || loading) return
    setMessages(m => [...m, { role: 'user', text: query }])
    setInput('')
    setLoading(true)
    setError('')
    try {
      const card = await askQuery(query, taskType)
      setMessages(m => [...m, { role: 'bot', card }])
      refreshStats()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function giveFeedback(index, thumbs) {
    try {
      await sendFeedback(messages[index].card.trace_id, thumbs)
      setMessages(m => m.map((msg, i) => (i === index ? { ...msg, feedback: thumbs } : msg)))
    } catch (e) {
      setError(e.message)
    }
  }

  function pickSample(e) {
    const s = samples[e.target.value]
    if (s) {
      setInput(s.query)
      setTaskType(s.task_type)
    }
    e.target.value = ''
  }

  return (
    <div className="page">
      <TopBar stats={stats} />
      <main className="chat">
        {messages.length === 0 && (
          <p className="empty">
            Ask anything. Each answer shows which model handled it, why, and what it cost.
            Pick a sample below to try the cache and escalation.
          </p>
        )}
        {messages.map((msg, i) =>
          msg.role === 'user' ? (
            <div key={i} className="bubble user">{msg.text}</div>
          ) : (
            <div key={i} className="bubble bot">
              <div className="answer">{msg.card.answer}</div>
              <RoutingCard card={msg.card} feedback={msg.feedback} onFeedback={t => giveFeedback(i, t)} />
            </div>
          ),
        )}
        {loading && <div className="bubble bot muted">Routing…</div>}
        <div ref={endRef} />
      </main>
      <footer className="composer">
        {error && <div className="error">{error}</div>}
        <div className="controls">
          <select defaultValue="" onChange={pickSample} aria-label="Sample questions">
            <option value="" disabled>Samples…</option>
            {samples.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
          <select value={taskType} onChange={e => setTaskType(e.target.value)} aria-label="Task type">
            <option value="">auto-detect type</option>
            <option value="mcq">mcq</option>
            <option value="math">math</option>
            <option value="text">text</option>
          </select>
        </div>
        <div className="row">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Type a question (Shift+Enter for a new line, e.g. for MCQ options)"
            rows={3}
          />
          <button className="send" onClick={send} disabled={loading || !input.trim()}>Send</button>
        </div>
      </footer>
    </div>
  )
}
