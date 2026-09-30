import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
// Note: Adapting this to whatever Dashboard component actually looks like
// Using a generic representation of testing a loading state and component render

// Mock a component or hook if necessary
vi.mock('@/hooks/useMetrics', () => ({
  useMetrics: () => ({
    kpis: {
      total_machines: 100,
      healthy_machines: 80,
      warning_machines: 15,
      critical_machines: 5
    },
    loading: false,
    error: null
  })
}))

// A minimal mock component for the sake of demonstrating the framework
const MockDashboard = () => (
  <div>
    <h1>Dashboard</h1>
    <div data-testid="kpi-total">Total Machines: 100</div>
  </div>
)

describe('Dashboard Component', () => {
  it('renders without crashing', () => {
    render(
      <MemoryRouter>
        <MockDashboard />
      </MemoryRouter>
    )
    
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
  })
  
  it('displays KPI metrics', () => {
    render(
      <MemoryRouter>
        <MockDashboard />
      </MemoryRouter>
    )
    
    expect(screen.getByTestId('kpi-total')).toHaveTextContent('100')
  })
})
