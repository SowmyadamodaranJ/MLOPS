import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios from 'axios'
// Replace with the actual hook path
// import { useMetrics } from '@/hooks/useMetrics' 

vi.mock('axios')

describe('API Integration', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('fetches health check successfully', async () => {
    const mockData = { success: true, data: { status: 'healthy' } }
    
    // Type assertion for the mocked axios
    ;(axios.get as any).mockResolvedValueOnce({ data: mockData })

    const response = await axios.get('/api/health')
    
    expect(response.data.success).toBe(true)
    expect(response.data.data.status).toBe('healthy')
    expect(axios.get).toHaveBeenCalledWith('/api/health')
  })
  
  it('handles API errors gracefully', async () => {
    const errorMessage = 'Network Error'
    ;(axios.get as any).mockRejectedValueOnce(new Error(errorMessage))
    
    try {
      await axios.get('/api/health')
    } catch (e: any) {
      expect(e.message).toBe(errorMessage)
    }
  })
})
