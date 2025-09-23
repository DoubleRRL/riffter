import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Share2, History, Zap, Mic, RefreshCw, Save, Edit3, MessageSquare } from 'lucide-react'
import './App.css'

function App() {
  const [topic, setTopic] = useState('')
  const [riffTopic, setRiffTopic] = useState('')
  const [currentRiff, setCurrentRiff] = useState('')
  const [currentJoke, setCurrentJoke] = useState({
    premise: '',
    punchline: '',
    initial_tag: '',
    alternate_angle: '',
    additional_tags: []
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [activeTab, setActiveTab] = useState('riff') // 'riff' or 'joke'
  const [editingField, setEditingField] = useState(null)

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('riffter-history')
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory))
      } catch (e) {
        console.error('Failed to load history:', e)
      }
    }
  }, [])

  // Save history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('riffter-history', JSON.stringify(history))
  }, [history])

  const generateRiff = async () => {
    const inputTopic = activeTab === 'riff' ? riffTopic : topic
    if (!inputTopic.trim()) {
      setError('Enter a topic or words to riff on!')
      return
    }

    setLoading(true)
    setError('')
    setCurrentRiff('')

    try {
      // Mock API call - replace with actual backend endpoint
      const response = await fetch('http://localhost:8000/riff', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: inputTopic.trim() }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      const newRiff = data.riff || 'Riff generation failed'

      setCurrentRiff(newRiff)

      // Add to history
      const historyItem = {
        id: Date.now(),
        type: 'riff',
        topic: inputTopic.trim(),
        content: newRiff,
        timestamp: new Date().toISOString()
      }
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]) // Keep last 10

    } catch (err) {
      setError(`Riff failed: ${err.message}`)
      console.error('Generation error:', err)
    } finally {
      setLoading(false)
    }
  }

  const generateJoke = async () => {
    if (!topic.trim()) {
      setError('Enter a topic for your joke!')
      return
    }

    setLoading(true)
    setError('')

    try {
      // Mock API call - replace with actual backend endpoint
      const response = await fetch('http://localhost:8000/joke', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic: topic.trim() }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      const jokeData = data.joke

      // Directly map the API response to the state
      const newJoke = {
        premise: jokeData.premise || '',
        punchline: jokeData.punchline || '',
        initial_tag: jokeData.initial_tag || '',
        alternate_angle: jokeData.alternate_angle || '',
        additional_tags: Array.isArray(jokeData.additional_tags) ? jokeData.additional_tags : []
      }

      setCurrentJoke(newJoke)

      // Add to history
      const historyItem = {
        id: Date.now(),
        type: 'joke',
        topic: topic.trim(),
        content: newJoke,
        timestamp: new Date().toISOString()
      }
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]) // Keep last 10

    } catch (err) {
      setError(`Joke failed: ${err.message}`)
      console.error('Generation error:', err)
    } finally {
      setLoading(false)
    }
  }

  const regenerateField = async (part, index = null) => {
    if (!topic.trim()) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/regenerate_joke_part', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: topic.trim(),
          joke_context: currentJoke,
          part_to_regenerate: part
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const newContent = data.new_content;

      setCurrentJoke(prev => {
        const updated = { ...prev };
        if (part === 'additional_tags') {
          // For tags, replace the entire array
          updated[part] = Array.isArray(newContent) ? newContent : [newContent];
        } else if (Object.keys(updated).includes(part)) {
          updated[part] = newContent;
        }
        return updated;
      });

    } catch (err) {
      setError(`Regeneration failed: ${err.message}`);
      console.error('Regeneration error:', err);
    } finally {
      setLoading(false);
    }
  };

  const updateJokeField = (fieldType, value, index = null) => {
    setCurrentJoke(prev => {
      const updated = { ...prev };
      if (fieldType === 'additional_tags' && index !== null) {
        updated.additional_tags = [...prev.additional_tags];
        updated.additional_tags[index] = value;
      } else if (Object.keys(updated).includes(fieldType)) {
        updated[fieldType] = value;
      }
      return updated;
    });
  };

  const saveJoke = () => {
    const jokeText = formatJokeForSharing(currentJoke, topic)
    const historyItem = {
      id: Date.now(),
      type: 'joke',
      topic: topic.trim(),
      content: currentJoke,
      timestamp: new Date().toISOString()
    }
    setHistory(prev => [historyItem, ...prev.slice(0, 9)])
    console.log('Joke saved to history!')
  }

  const formatJokeForSharing = (joke, topicText) => {
    let text = `🎭 Joke about "${topicText}":\n\n`
    if (joke.premise) text += `Premise: ${joke.premise}\n\n`
    if (joke.punchline) text += `Punchline: ${joke.punchline}\n`
    if (joke.initial_tag) text += `Initial Tag: ${joke.initial_tag}\n`
    if (joke.alternate_angle) text += `\nAnother Angle: ${joke.alternate_angle}\n`
    if (joke.additional_tags.some(tag => tag.trim())) {
      text += `\nTags:\n`
      joke.additional_tags.forEach((tag, i) => {
        if (tag.trim()) text += `• ${tag}\n`
      })
    }
    return text
  }

  const shareContent = async (content, topicText, type = 'riff') => {
    const shareText = type === 'riff'
      ? `🎭 Comedy riff on "${topicText}":\n\n${content}\n\n#Riffter #ComedyAI`
      : formatJokeForSharing(content, topicText)

    if (navigator.share) {
      try {
        await navigator.share({
          title: type === 'riff' ? 'Riffter Riff' : 'Riffter Joke',
          text: shareText,
        })
      } catch (e) {
        // User cancelled or share failed, fallback to clipboard
        copyToClipboard(shareText)
      }
    } else {
      copyToClipboard(shareText)
    }
  }

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      // Could add a toast notification here
      console.log('Copied to clipboard!')
    } catch (e) {
      console.error('Failed to copy:', e)
    }
  }

  const loadFromHistory = (historyItem) => {
    if (historyItem.type === 'riff') {
      setActiveTab('riff')
      setRiffTopic(historyItem.topic)
      setCurrentRiff(historyItem.content)
    } else {
      setActiveTab('joke')
      setTopic(historyItem.topic)
      setCurrentJoke(historyItem.content)
    }
    setShowHistory(false)
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem('riffter-history')
  }

  return (
    <div className="bouncy-app">
      {/* Animated Background */}
      <div className="bg-animation">
        <div className="floating-shapes">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="shape"
              animate={{
                y: [0, -20, 0],
                rotate: [0, 180, 360],
              }}
              transition={{
                duration: 3 + i,
                repeat: Infinity,
                ease: "easeInOut",
              }}
              style={{
                left: `${20 + i * 15}%`,
                animationDelay: `${i * 0.5}s`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Header */}
      <motion.header
        className="bouncy-header"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <motion.div
          className="logo"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Mic className="logo-icon" />
          <h1>Riffter</h1>
        </motion.div>
        <p>AI-Powered Comedy Generation</p>
      </motion.header>

      <main className="bouncy-main">
        {/* Tab Navigation */}
        <motion.div
          className="tab-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
        >
          <div className="tab-buttons">
            <motion.button
              className={`tab-btn ${activeTab === 'riff' ? 'active' : ''}`}
              onClick={() => setActiveTab('riff')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <MessageSquare className="tab-icon" />
              Riff Bot
            </motion.button>
            <motion.button
              className={`tab-btn ${activeTab === 'joke' ? 'active' : ''}`}
              onClick={() => setActiveTab('joke')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Mic className="tab-icon" />
              Full Joke
            </motion.button>
          </div>
        </motion.div>

        {/* Input Section */}
        <motion.div
          className="input-container"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          {activeTab === 'riff' ? (
            <div className="input-wrapper">
              <motion.input
                type="text"
                placeholder="Enter words or a topic to riff on..."
                value={riffTopic}
                onChange={(e) => setRiffTopic(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !loading && generateRiff()}
                className="topic-input"
                whileFocus={{ scale: 1.02 }}
                disabled={loading}
              />
              <motion.button
                onClick={generateRiff}
                disabled={loading}
                className="generate-btn"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                animate={loading ? { rotate: 360 } : {}}
                transition={loading ? { duration: 1, repeat: Infinity, ease: "linear" } : {}}
              >
                {loading ? <Zap className="spin" /> : 'Generate Riff'}
              </motion.button>
            </div>
          ) : (
            <div className="input-wrapper">
              <motion.input
                type="text"
                placeholder="Enter a topic for your joke..."
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !loading && generateJoke()}
                className="topic-input"
                whileFocus={{ scale: 1.02 }}
                disabled={loading}
              />
              <motion.button
                onClick={generateJoke}
                disabled={loading}
                className="generate-btn"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                animate={loading ? { rotate: 360 } : {}}
                transition={loading ? { duration: 1, repeat: Infinity, ease: "linear" } : {}}
              >
                {loading ? <Zap className="spin" /> : 'Generate Joke'}
              </motion.button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons">
            <motion.button
              onClick={() => setShowHistory(!showHistory)}
              className="history-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <History />
              History ({history.length})
            </motion.button>
          </div>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="error-message"
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content Output */}
        <AnimatePresence mode="wait">
          {activeTab === 'riff' && currentRiff && (
            <motion.div
              key="riff"
              className="content-container"
              initial={{ scale: 0.8, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.8, opacity: 0, y: -20 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              <div className="content-header">
                <h3>Comedy Riff on "{riffTopic}"</h3>
                <div className="content-actions">
                  <motion.button
                    onClick={() => shareContent(currentRiff, riffTopic, 'riff')}
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Share2 />
                  </motion.button>
                  <motion.button
                    onClick={generateRiff}
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <RefreshCw />
                  </motion.button>
                </div>
              </div>
              <motion.div
                className="content-body"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                {currentRiff.split('\n').map((line, index) => (
                  <motion.p
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 * index }}
                  >
                    {line}
                  </motion.p>
                ))}
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'joke' && (currentJoke.premise || currentJoke.punchline || currentJoke.initial_tag) && (
            <motion.div
              key="joke"
              className="joke-container"
              initial={{ scale: 0.8, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.8, opacity: 0, y: -20 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              <div className="joke-header">
                <h3>Joke about "{topic}"</h3>
                <div className="joke-actions">
                  <motion.button
                    onClick={saveJoke}
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Save />
                  </motion.button>
                  <motion.button
                    onClick={() => shareContent(currentJoke, topic, 'joke')}
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Share2 />
                  </motion.button>
                  <motion.button
                    onClick={generateJoke}
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <RefreshCw />
                  </motion.button>
                </div>
              </div>

              <div className="joke-structure">
                {/* Premise */}
                <div className="joke-field">
                  <div className="field-header">
                    <span className="field-label">Premise</span>
                    <motion.button
                      onClick={() => regenerateField('premise')}
                      className="regen-btn"
                      whileHover={{ scale: 1.1, rotate: 15 }}
                      whileTap={{ scale: 0.9, rotate: -15, background: '#2ecc71' }}
                    >
                      <RefreshCw />
                    </motion.button>
                  </div>
                  {editingField === 'premise' ? (
                    <textarea
                      value={currentJoke.premise}
                      onChange={(e) => updateJokeField('premise', e.target.value)}
                      onBlur={() => setEditingField(null)}
                      autoFocus
                      className="joke-input"
                    />
                  ) : (
                    <div
                      className="joke-text"
                      onClick={() => setEditingField('premise')}
                    >
                      {currentJoke.premise || "Click to add premise..."}
                    </div>
                  )}
                </div>

                {/* Punchline */}
                <div className="joke-field">
                  <div className="field-header">
                    <span className="field-label">Punchline</span>
                    <motion.button
                      onClick={() => regenerateField('punchline')}
                      className="regen-btn"
                      whileHover={{ scale: 1.1, rotate: 15 }}
                      whileTap={{ scale: 0.9, rotate: -15, background: '#2ecc71' }}
                    >
                      <RefreshCw />
                    </motion.button>
                  </div>
                  {editingField === 'punchline' ? (
                    <textarea
                      value={currentJoke.punchline}
                      onChange={(e) => updateJokeField('punchline', e.target.value)}
                      onBlur={() => setEditingField(null)}
                      autoFocus
                      className="joke-input"
                    />
                  ) : (
                    <div
                      className="joke-text"
                      onClick={() => setEditingField('punchline')}
                    >
                      {currentJoke.punchline || "Click to add punchline..."}
                    </div>
                  )}
                </div>

                {/* Initial Tag */}
                <div className="joke-field">
                  <div className="field-header">
                    <span className="field-label">Initial Tag</span>
                    <motion.button
                      onClick={() => regenerateField('initial_tag')}
                      className="regen-btn"
                      whileHover={{ scale: 1.1, rotate: 15 }}
                      whileTap={{ scale: 0.9, rotate: -15, background: '#2ecc71' }}
                    >
                      <RefreshCw />
                    </motion.button>
                  </div>
                  {editingField === 'initial_tag' ? (
                    <textarea
                      value={currentJoke.initial_tag}
                      onChange={(e) => updateJokeField('initial_tag', e.target.value)}
                      onBlur={() => setEditingField(null)}
                      autoFocus
                      className="joke-input"
                    />
                  ) : (
                    <div
                      className="joke-text"
                      onClick={() => setEditingField('initial_tag')}
                    >
                      {currentJoke.initial_tag || "Click to add initial tag..."}
                    </div>
                  )}
                </div>

                {/* Another Angle */}
                <div className="joke-field">
                  <div className="field-header">
                    <span className="field-label">Another Angle</span>
                    <motion.button
                      onClick={() => regenerateField('alternate_angle')}
                      className="regen-btn"
                      whileHover={{ scale: 1.1, rotate: 15 }}
                      whileTap={{ scale: 0.9, rotate: -15, background: '#2ecc71' }}
                    >
                      <RefreshCw />
                    </motion.button>
                  </div>
                  {editingField === 'alternate_angle' ? (
                    <textarea
                      value={currentJoke.alternate_angle}
                      onChange={(e) => updateJokeField('alternate_angle', e.target.value)}
                      onBlur={() => setEditingField(null)}
                      autoFocus
                      className="joke-input"
                    />
                  ) : (
                    <div
                      className="joke-text"
                      onClick={() => setEditingField('alternate_angle')}
                    >
                      {currentJoke.alternate_angle || "Click to add another angle..."}
                    </div>
                  )}
                </div>

                {/* Tags */}
                {currentJoke.additional_tags.map((tag, index) => (
                  <div key={index} className="joke-field">
                    <div className="field-header">
                      <span className="field-label">Tag {index + 1}</span>
                      <motion.button
                        // Note: regenerating one tag regenerates all of them per backend logic
                        onClick={() => regenerateField('additional_tags', index)}
                        className="regen-btn"
                        whileHover={{ scale: 1.1, rotate: 15 }}
                        whileTap={{ scale: 0.9, rotate: -15, background: '#2ecc71' }}
                      >
                        <RefreshCw />
                      </motion.button>
                    </div>
                    {editingField === `tag-${index}` ? (
                      <input
                        type="text"
                        value={tag}
                        onChange={(e) => updateJokeField('additional_tags', e.target.value, index)}
                        onBlur={() => setEditingField(null)}
                        autoFocus
                        className="tag-input"
                      />
                    ) : (
                      <div
                        className="tag-text"
                        onClick={() => setEditingField(`tag-${index}`)}
                      >
                        {tag || "Click to add tag..."}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* History Panel */}
        <AnimatePresence>
          {showHistory && (
            <motion.div
              className="history-panel"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="history-header">
                <h4>Previous Generations</h4>
                <button onClick={clearHistory} className="clear-btn">
                  Clear All
                </button>
              </div>
              <div className="history-list">
                {history.length === 0 ? (
                  <p className="empty-history">No generations yet. Create your first riff or joke!</p>
                ) : (
                  history.map((item) => (
                    <motion.div
                      key={item.id}
                      className="history-item"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      whileHover={{ scale: 1.02 }}
                      onClick={() => loadFromHistory(item)}
                    >
                      <div className="history-topic">{item.topic}</div>
                      <div className="history-type">{item.type === 'riff' ? '🎭 Riff' : '🎪 Joke'}</div>
                      <div className="history-preview">
                        {item.type === 'riff'
                          ? item.content.substring(0, 100)
                          : item.content.premise?.substring(0, 100) || 'Structured joke'
                        }...
                      </div>
                      <div className="history-date">
                        {new Date(item.timestamp).toLocaleDateString()}
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
