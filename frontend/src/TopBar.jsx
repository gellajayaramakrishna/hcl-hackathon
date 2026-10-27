// Top bar: live spend vs "if always large", savings %, cache hit rate, and a routing-mix bar chart.
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis } from 'recharts'
import { money } from './api.js'

function MixTooltip({ active, payload, total }) {
  if (!active || !payload?.length) return null
  const { name, count } = payload[0].payload
  const pct = total ? Math.round((100 * count) / total) : 0
  return (
    <div className="tooltip">
      <strong>{name}</strong>: {count} request{count === 1 ? '' : 's'} ({pct}%)
    </div>
  )
}

export default function TopBar({ stats }) {
  if (!stats) return <header className="topbar">Connecting to backend…</header>
  const cur = stats.currency
  const mix = Object.entries(stats.routing_mix).map(([name, count]) => ({ name, count }))
  return (
    <header className="topbar">
      <div className="brand">
        <h1>Cost-Aware Model Router</h1>
        {stats.mock_mode && <span className="badge">MOCK MODE</span>}
      </div>
      <div className="kpis">
        <div className="kpi">
          <span className="kpi-label">Spend</span>
          <span className="kpi-value">{money(stats.total_spend, cur)}</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">If always large</span>
          <span className="kpi-value muted">{money(stats.if_always_large, cur)}</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Savings</span>
          <span className="kpi-value good">{stats.savings_pct.toFixed(1)}%</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Cache hits</span>
          <span className="kpi-value">{(100 * stats.cache_hit_rate).toFixed(0)}%</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Requests</span>
          <span className="kpi-value">{stats.requests}</span>
        </div>
      </div>
      <figure className="mix">
        <figcaption>Routing mix</figcaption>
        <ResponsiveContainer width="100%" height={92}>
          <BarChart data={mix} margin={{ top: 16, right: 4, bottom: 0, left: 4 }} barCategoryGap="25%">
            <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: 'var(--border)' }}
                   tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} interval={0} />
            <Tooltip content={<MixTooltip total={stats.requests} />} cursor={{ fill: 'var(--hover)' }} />
            <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              <LabelList dataKey="count" position="top" fill="var(--text-secondary)" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </figure>
    </header>
  )
}
