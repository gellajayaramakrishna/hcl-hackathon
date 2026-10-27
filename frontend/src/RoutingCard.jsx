// The "routing card" under each answer: model, why, cache, confidence, escalations, cost, time,
// per-step latency, and 👍 / 👎 feedback buttons.
import { money } from './api.js'

const CACHE_LABEL = { none: 'miss', exact: 'exact hit', similar: 'similar hit' }

export default function RoutingCard({ card, feedback, onFeedback }) {
  return (
    <div className="card">
      <div className="card-grid">
        <span className="k">Model</span>
        <span className="v"><span className={`model model-${card.model_used}`}>{card.model_used}</span></span>
        <span className="k">Why</span>
        <span className="v why">{card.why}</span>
        <span className="k">Cache</span>
        <span className="v">{CACHE_LABEL[card.cache] ?? card.cache}</span>
        <span className="k">Confidence</span>
        <span className="v">{card.confidence}</span>
        <span className="k">Escalations</span>
        <span className="v">{card.escalations}</span>
        <span className="k">Cost</span>
        <span className="v">{money(card.cost, card.currency)}</span>
        <span className="k">Time</span>
        <span className="v">{Math.round(card.total_ms)} ms</span>
      </div>
      <details>
        <summary>Latency per step</summary>
        <ul className="steps">
          {Object.entries(card.latency_ms).map(([step, ms]) => (
            <li key={step}><span>{step}</span><span>{ms} ms</span></li>
          ))}
        </ul>
        <div className="trace">trace {card.trace_id}</div>
      </details>
      <div className="feedback">
        {feedback ? (
          <span className="muted">{feedback === 1 ? '👍' : '👎'} Thanks for the feedback</span>
        ) : (
          <>
            <button onClick={() => onFeedback(1)} aria-label="Good answer">👍</button>
            <button onClick={() => onFeedback(-1)} aria-label="Bad answer">👎</button>
          </>
        )}
      </div>
    </div>
  )
}
