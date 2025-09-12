import { useState } from 'react'
import './App.css'

function App() {
  const [mode, setMode] = useState('riff') // 'riff' or 'joke'
  const [topic, setTopic] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const generateContent = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic')
      return
    }

    setLoading(true)
    setError('')
    setOutput('')

    try {
      const endpoint = mode === 'riff' ? '/riff' : '/joke'
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topic.trim() }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (mode === 'riff') {
        setOutput(data.riff)
      } else {
        const joke = data.joke
        setOutput(`${joke.premise}\n\n${joke.punchline}\n\n${joke.initial_tag}\n\n${joke.alternate_angle}\n\n${joke.additional_tags.join('\n')}`)
      }
    } catch (err) {
      setError(`Failed to generate ${mode}: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Riffter</h1>
        <p>AI-powered comedy riff generator</p>
      </header>

      <main>
        <div className="mode-toggle">
          <button
            className={mode === 'riff' ? 'active' : ''}
            onClick={() => setMode('riff')}
          >
            Riff Mode
          </button>
          <button
            className={mode === 'joke' ? 'active' : ''}
            onClick={() => setMode('joke')}
          >
            Joke Mode
          </button>
        </div>

        <div className="input-section">
          <input
            type="text"
            placeholder={`Enter a topic for ${mode} generation...`}
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && generateContent()}
          />
          <button onClick={generateContent} disabled={loading}>
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {output && (
          <div className="output-section">
            <h3>Generated {mode === 'riff' ? 'Riff' : 'Joke'}:</h3>
            <div className="output">
              {output.split('\n').map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
            <button onClick={generateContent} disabled={loading}>
              Regenerate
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
