// Small wrappers around the FastAPI backend (POST /query, POST /feedback, GET /stats, GET /samples).
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function call(path, body) {
  const res = await fetch(BASE + path, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`)
  return res.json()
}

export const askQuery = (query, taskType) => call('/query', { query, task_type: taskType || null })
export const sendFeedback = (traceId, thumbs) => call('/feedback', { trace_id: traceId, thumbs })
export const getStats = () => call('/stats')
export const getSamples = () => call('/samples')

const SYMBOLS = { USD: '$', INR: '₹', EUR: '€', GBP: '£' }

// tiny costs need more decimals: 0.000054 -> "$0.000054"
export function money(value, currency) {
  const symbol = SYMBOLS[currency] ?? `${currency} `
  const digits = value === 0 ? 2 : value < 0.01 ? 6 : value < 1 ? 4 : 2
  return symbol + value.toFixed(digits)
}
