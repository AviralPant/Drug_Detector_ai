import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, Terminal, Zap, Cpu, Activity, Code, Lock, Unlock } from 'lucide-react';

const VoiceAIAssistant = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [language, setLanguage] = useState('en-IN');
  const [history, setHistory] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [terminalLines, setTerminalLines] = useState([]);
  const [glitchText, setGlitchText] = useState('INITIALIZING...');
  const [systemStatus, setSystemStatus] = useState('ONLINE');
  const recognitionRef = useRef(null);
  const synthRef = useRef(window.speechSynthesis);

  // Matrix rain effect
  useEffect(() => {
    const canvas = document.getElementById('matrix-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = '01アイウエオカキクケコサシスセソタチツテト';
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = Array(Math.floor(columns)).fill(1);

    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#0F0';
      ctx.font = fontSize + 'px monospace';

      for (let i = 0; i < drops.length; i++) {
        const text = chars[Math.floor(Math.random() * chars.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };

    const interval = setInterval(draw, 33);
    return () => clearInterval(interval);
  }, []);

  // Glitch effect
  useEffect(() => {
    const glitchInterval = setInterval(() => {
      const texts = ['VOICE_AI_ACTIVE', 'NEURAL_LINK_ESTABLISHED', 'SYSTEM_READY', 'AWAITING_INPUT'];
      setGlitchText(texts[Math.floor(Math.random() * texts.length)]);
    }, 3000);
    return () => clearInterval(glitchInterval);
  }, []);

  useEffect(() => {
    // Initialize speech synthesis on component mount
    if (window.speechSynthesis) {
      // Load voices
      window.speechSynthesis.getVoices();
      
      // Speak welcome message
      setTimeout(() => {
        const welcomeMsg = language === 'hi-IN' 
          ? 'नमस्ते। वॉयस ए आई सिस्टम एक्टिवेट हुआ। माइक बटन दबाएं और कमांड दें।'
          : 'Hello. Voice AI System activated. Press the mic button and speak your command.';
        speak(welcomeMsg, language);
        addTerminalLine('[SYSTEM] WELCOME_MESSAGE_PLAYED', 'success');
      }, 1000);
    }

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = language;

      recognitionRef.current.onresult = (event) => {
        const text = event.results[0][0].transcript;
        setTranscript(text);
        addTerminalLine(`> VOICE_INPUT: ${text}`, 'input');
        processCommand(text);
      };

      recognitionRef.current.onerror = (event) => {
        addTerminalLine(`[ERROR] RECOGNITION_FAILED: ${event.error}`, 'error');
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        setSystemStatus('STANDBY');
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [language]);

  const addTerminalLine = (text, type = 'normal') => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setTerminalLines(prev => [...prev, { text, type, timestamp }].slice(-50));
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      addTerminalLine('[SYSTEM] MIC_DEACTIVATED', 'system');
    } else {
      recognitionRef.current.lang = language;
      recognitionRef.current?.start();
      setIsListening(true);
      setSystemStatus('LISTENING');
      setTranscript('');
      setResponse('');
      addTerminalLine('[SYSTEM] MIC_ACTIVATED :: AWAITING_VOICE_INPUT', 'system');
    }
  };

  const speak = (text, lang) => {
    if (synthRef.current && text) {
      // Cancel any ongoing speech
      synthRef.current.cancel();
      
      // Small delay to ensure cancellation is complete
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang;
        utterance.rate = 1.1;
        utterance.pitch = 0.9;
        utterance.volume = 1.0; // Maximum volume
        
        utterance.onstart = () => {
          setIsSpeaking(true);
          setSystemStatus('SPEAKING');
          addTerminalLine('[AUDIO_OUTPUT] VOICE_SYNTHESIS_ACTIVE', 'success');
        };
        utterance.onend = () => {
          setIsSpeaking(false);
          setSystemStatus('ONLINE');
          addTerminalLine('[AUDIO_OUTPUT] VOICE_SYNTHESIS_COMPLETE', 'system');
        };
        utterance.onerror = (event) => {
          console.error('Speech synthesis error:', event);
          addTerminalLine('[AUDIO_OUTPUT] ERROR: SYNTHESIS_FAILED', 'error');
          setIsSpeaking(false);
        };
        
        synthRef.current.speak(utterance);
      }, 100);
    }
  };

  const processCommand = (command) => {
    const lowerCommand = command.toLowerCase();
    let responseText = '';
    let detectedLang = language;

    addTerminalLine('[NEURAL_NET] PROCESSING_COMMAND...', 'processing');

    // Time commands
    if (lowerCommand.includes('time') || lowerCommand.includes('समय') || lowerCommand.includes('टाइम')) {
      const now = new Date();
      const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      responseText = language === 'hi-IN' 
        ? `सिस्टम टाइम ${timeStr} है`
        : `System time is ${timeStr}`;
      addTerminalLine(`[TIME_MODULE] ${timeStr}`, 'success');
    }
    
    // Date commands
    else if (lowerCommand.includes('date') || lowerCommand.includes('तारीख') || lowerCommand.includes('डेट')) {
      const now = new Date();
      const dateStr = now.toLocaleDateString('en-IN', { year: 'numeric', month: '2-digit', day: '2-digit' });
      responseText = language === 'hi-IN'
        ? `सिस्टम डेट ${dateStr} है`
        : `System date is ${dateStr}`;
      addTerminalLine(`[DATE_MODULE] ${dateStr}`, 'success');
    }
    
    // Calculator commands
    else if (lowerCommand.includes('calculate') || lowerCommand.includes('गणना') || lowerCommand.includes('calculate करो')) {
      try {
        const mathExpr = lowerCommand.replace(/calculate|गणना|करो|कैलकुलेट/g, '').trim();
        const numbers = mathExpr.match(/\d+/g);
        if (numbers && numbers.length >= 2) {
          let result = 0;
          let operation = '';
          if (mathExpr.includes('+') || mathExpr.includes('plus') || mathExpr.includes('जोड़')) {
            result = numbers.reduce((a, b) => parseInt(a) + parseInt(b), 0);
            operation = 'ADD';
          } else if (mathExpr.includes('-') || mathExpr.includes('minus') || mathExpr.includes('घटा')) {
            result = parseInt(numbers[0]) - parseInt(numbers[1]);
            operation = 'SUBTRACT';
          } else if (mathExpr.includes('*') || mathExpr.includes('multiply') || mathExpr.includes('गुणा')) {
            result = numbers.reduce((a, b) => parseInt(a) * parseInt(b), 1);
            operation = 'MULTIPLY';
          }
          responseText = language === 'hi-IN'
            ? `कंप्यूटेशन कम्पलीट। रिजल्ट ${result}`
            : `Computation complete. Result is ${result}`;
          addTerminalLine(`[CALC_ENGINE] ${operation} :: OUTPUT=${result}`, 'success');
        }
      } catch (e) {
        responseText = language === 'hi-IN'
          ? 'कंप्यूटेशन एरर'
          : 'Computation error';
        addTerminalLine('[CALC_ENGINE] ERROR', 'error');
      }
    }
    
    // Search commands
    else if (lowerCommand.includes('search') || lowerCommand.includes('खोज') || lowerCommand.includes('सर्च')) {
      const query = lowerCommand.replace(/search|खोज|सर्च|करो|for/g, '').trim();
      responseText = language === 'hi-IN'
        ? `सर्च इंजन एक्टिवेट हो रहा है ${query} के लिए`
        : `Activating search engine for ${query}`;
      addTerminalLine(`[SEARCH_ENGINE] QUERY="${query}" :: LAUNCHING_BROWSER`, 'success');
      setTimeout(() => {
        window.open(`https://www.google.com/search?q=${encodeURIComponent(query)}`, '_blank');
      }, 1000);
    }
    
    // Open applications
    else if (lowerCommand.includes('open') || lowerCommand.includes('खोल') || lowerCommand.includes('ओपन')) {
      if (lowerCommand.includes('youtube')) {
        responseText = language === 'hi-IN' ? 'YouTube प्रोटोकॉल एक्सीक्यूट हो रहा है' : 'Executing YouTube protocol';
        addTerminalLine('[APP_LAUNCHER] TARGET=youtube.com :: STATUS=LAUNCHING', 'success');
        setTimeout(() => window.open('https://youtube.com', '_blank'), 500);
      } else if (lowerCommand.includes('gmail') || lowerCommand.includes('email')) {
        responseText = language === 'hi-IN' ? 'Gmail सिस्टम एक्सेस हो रहा है' : 'Accessing Gmail system';
        addTerminalLine('[APP_LAUNCHER] TARGET=gmail.com :: STATUS=LAUNCHING', 'success');
        setTimeout(() => window.open('https://gmail.com', '_blank'), 500);
      } else if (lowerCommand.includes('github')) {
        responseText = language === 'hi-IN' ? 'GitHub रिपोजिटरी एक्सेस हो रहा है' : 'Accessing GitHub repository';
        addTerminalLine('[APP_LAUNCHER] TARGET=github.com :: STATUS=LAUNCHING', 'success');
        setTimeout(() => window.open('https://github.com', '_blank'), 500);
      } else {
        responseText = language === 'hi-IN'
          ? 'टार्गेट एप्लिकेशन स्पेसिफाई करें'
          : 'Specify target application';
        addTerminalLine('[APP_LAUNCHER] ERROR: TARGET_UNDEFINED', 'error');
      }
    }
    
    // Weather
    else if (lowerCommand.includes('weather') || lowerCommand.includes('मौसम')) {
      responseText = language === 'hi-IN'
        ? 'वेदर डेटा फेच हो रहा है'
        : 'Fetching weather data';
      addTerminalLine('[WEATHER_API] FETCHING_DATA...', 'processing');
      setTimeout(() => window.open('https://www.google.com/search?q=weather', '_blank'), 500);
    }
    
    // Greeting
    else if (lowerCommand.includes('hello') || lowerCommand.includes('hi') || lowerCommand.includes('नमस्ते') || lowerCommand.includes('हेलो')) {
      responseText = language === 'hi-IN'
        ? 'नमस्ते। न्यूरल सिस्टम ऑनलाइन है। कमांड दें।'
        : 'Hello. Neural system online. Awaiting command.';
      addTerminalLine('[GREETING_PROTOCOL] USER_ACKNOWLEDGED', 'success');
    }
    
    // Hack mode easter egg
    else if (lowerCommand.includes('hack') || lowerCommand.includes('हैक')) {
      responseText = language === 'hi-IN'
        ? 'हैकिंग प्रोटोकॉल एक्टिवेट। एक्सेस ग्रांटेड।'
        : 'Hacking protocol activated. Access granted.';
      addTerminalLine('[SECURITY] FIREWALL_BYPASSED :: ROOT_ACCESS_GRANTED', 'success');
      addTerminalLine('[KERNEL] INJECTING_PAYLOAD...', 'processing');
      setTimeout(() => addTerminalLine('[SUCCESS] SYSTEM_COMPROMISED', 'success'), 1000);
    }
    
    // Default
    else {
      responseText = language === 'hi-IN'
        ? 'कमांड नॉट रिकॉग्नाइज्ड। रिट्राई करें।'
        : 'Command not recognized. Please retry.';
      addTerminalLine('[ERROR] UNKNOWN_COMMAND', 'error');
    }

    setResponse(responseText);
    speak(responseText, language);
    addTerminalLine(`< RESPONSE: ${responseText}`, 'output');
    
    setHistory(prev => [{
      command: command,
      response: responseText,
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
      language: language
    }, ...prev.slice(0, 9)]);
  };

  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      {/* Matrix Background */}
      <canvas id="matrix-canvas" className="fixed inset-0 opacity-20" />
      
      {/* Scanlines */}
      <div className="fixed inset-0 pointer-events-none bg-gradient-to-b from-transparent via-green-500/5 to-transparent animate-pulse" 
           style={{ backgroundSize: '100% 4px' }} />

      <div className="relative z-10 p-4 max-w-7xl mx-auto">
        {/* Header */}
        <div className="border-2 border-green-500 bg-black/90 backdrop-blur-sm mb-4 p-4 font-mono shadow-lg shadow-green-500/50">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <Terminal className="w-6 h-6 text-green-500 animate-pulse" />
              <div>
                <h1 className="text-2xl font-bold text-green-500 tracking-wider glitch" data-text="VOICE_AI_SYSTEM">
                  VOICE_AI_SYSTEM
                </h1>
                <div className="text-xs text-green-400 mt-1 flex items-center gap-2">
                  <Activity className="w-3 h-3" />
                  <span>STATUS: {systemStatus}</span>
                  <span className="ml-4">LANG: {language === 'en-IN' ? 'EN' : 'HI'}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-green-500 animate-spin" style={{ animationDuration: '3s' }} />
              <Code className="w-5 h-5 text-green-500" />
              {isListening ? (
                <Unlock className="w-5 h-5 text-red-500 animate-pulse" />
              ) : (
                <Lock className="w-5 h-5 text-green-500" />
              )}
            </div>
          </div>

          <div className="flex gap-2 text-xs">
            <button
              onClick={() => setLanguage('en-IN')}
              className={`px-3 py-1 border transition-all font-bold tracking-wider ${
                language === 'en-IN'
                  ? 'border-green-500 bg-green-500/20 text-green-400 shadow-lg shadow-green-500/50'
                  : 'border-green-700 text-green-700 hover:border-green-500 hover:text-green-500'
              }`}
            >
              [EN]
            </button>
            <button
              onClick={() => setLanguage('hi-IN')}
              className={`px-3 py-1 border transition-all font-bold tracking-wider ${
                language === 'hi-IN'
                  ? 'border-green-500 bg-green-500/20 text-green-400 shadow-lg shadow-green-500/50'
                  : 'border-green-700 text-green-700 hover:border-green-500 hover:text-green-500'
              }`}
            >
              [HI]
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left Panel - Voice Control */}
          <div className="space-y-4">
            {/* Main Control */}
            <div className="border-2 border-green-500 bg-black/90 backdrop-blur-sm p-6 font-mono shadow-lg shadow-green-500/50">
              <div className="flex flex-col items-center">
                <div className="relative mb-4">
                  <button
                    onClick={toggleListening}
                    className={`w-40 h-40 border-4 rounded-full flex items-center justify-center transition-all font-mono relative ${
                      isListening
                        ? 'border-red-500 bg-red-500/10 shadow-2xl shadow-red-500/50 animate-pulse'
                        : 'border-green-500 bg-green-500/5 shadow-2xl shadow-green-500/50 hover:bg-green-500/10'
                    }`}
                  >
                    {isListening ? (
                      <>
                        <MicOff className="w-16 h-16 text-red-500" />
                        <div className="absolute inset-0 border-4 border-red-500 rounded-full animate-ping opacity-75" />
                      </>
                    ) : (
                      <Mic className="w-16 h-16 text-green-500" />
                    )}
                  </button>
                </div>
                
                <div className="text-center mb-4">
                  <p className="text-green-500 text-sm font-bold tracking-widest mb-2">
                    {glitchText}
                  </p>
                  <p className="text-green-400 text-xs">
                    {isListening ? '[RECORDING_ACTIVE]' : '[CLICK_TO_ACTIVATE]'}
                  </p>
                </div>

                {isSpeaking && (
                  <div className="flex items-center gap-2 text-green-400 border border-green-500 px-3 py-1 bg-green-500/10">
                    <Volume2 className="w-4 h-4 animate-pulse" />
                    <span className="text-xs">[AUDIO_OUTPUT_ACTIVE]</span>
                  </div>
                )}
              </div>

              {transcript && (
                <div className="mt-4 border border-green-500 bg-green-500/5 p-3">
                  <p className="text-green-600 text-xs mb-1">&gt;&gt; INPUT:</p>
                  <p className="text-green-400 text-sm font-bold">{transcript}</p>
                </div>
              )}

              {response && (
                <div className="mt-3 border border-cyan-500 bg-cyan-500/5 p-3">
                  <p className="text-cyan-600 text-xs mb-1">&lt;&lt; OUTPUT:</p>
                  <p className="text-cyan-400 text-sm font-bold">{response}</p>
                </div>
              )}
            </div>

            {/* Command Reference */}
            <div className="border-2 border-green-500 bg-black/90 backdrop-blur-sm p-4 font-mono shadow-lg shadow-green-500/50">
              <div className="flex items-center gap-2 mb-3 text-green-500">
                <Zap className="w-4 h-4" />
                <h3 className="text-sm font-bold tracking-wider">COMMAND_REFERENCE</h3>
              </div>
              <div className="space-y-1 text-xs text-green-400 max-h-48 overflow-y-auto">
                {language === 'hi-IN' ? (
                  <>
                    <div className="hover:bg-green-500/10 p-1">• समय बताओ</div>
                    <div className="hover:bg-green-500/10 p-1">• आज की तारीख</div>
                    <div className="hover:bg-green-500/10 p-1">• YouTube खोलो</div>
                    <div className="hover:bg-green-500/10 p-1">• 10 प्लस 20 कैलकुलेट करो</div>
                    <div className="hover:bg-green-500/10 p-1">• मौसम की जानकारी</div>
                    <div className="hover:bg-green-500/10 p-1">• Gmail खोलो</div>
                    <div className="hover:bg-green-500/10 p-1">• GitHub खोलो</div>
                  </>
                ) : (
                  <>
                    <div className="hover:bg-green-500/10 p-1">• time</div>
                    <div className="hover:bg-green-500/10 p-1">• date</div>
                    <div className="hover:bg-green-500/10 p-1">• open youtube</div>
                    <div className="hover:bg-green-500/10 p-1">• calculate 10 plus 20</div>
                    <div className="hover:bg-green-500/10 p-1">• weather</div>
                    <div className="hover:bg-green-500/10 p-1">• open gmail</div>
                    <div className="hover:bg-green-500/10 p-1">• open github</div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Terminal */}
          <div className="space-y-4">
            {/* Live Terminal */}
            <div className="border-2 border-green-500 bg-black/90 backdrop-blur-sm font-mono shadow-lg shadow-green-500/50 h-[600px] flex flex-col">
              <div className="border-b border-green-500 p-2 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-green-500" />
                <span className="text-green-500 text-xs font-bold tracking-wider">SYSTEM_TERMINAL</span>
                <div className="ml-auto flex gap-1">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                </div>
              </div>
              <div className="flex-1 p-3 overflow-y-auto space-y-1 text-xs">
                {terminalLines.map((line, idx) => (
                  <div key={idx} className={`flex gap-2 ${
                    line.type === 'error' ? 'text-red-500' :
                    line.type === 'success' ? 'text-green-400' :
                    line.type === 'processing' ? 'text-yellow-400' :
                    line.type === 'input' ? 'text-cyan-400' :
                    line.type === 'output' ? 'text-purple-400' :
                    'text-green-500'
                  }`}>
                    <span className="text-green-600">[{line.timestamp}]</span>
                    <span>{line.text}</span>
                  </div>
                ))}
                {terminalLines.length === 0 && (
                  <div className="text-green-600">
                    <p>[SYSTEM] TERMINAL_INITIALIZED</p>
                    <p>[SYSTEM] AWAITING_COMMANDS...</p>
                    <p className="animate-pulse">_</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes glitch {
          0%, 100% { text-shadow: 0 0 10px #0f0, 0 0 20px #0f0; }
          25% { text-shadow: -2px 0 10px #0f0, 2px 2px 20px #0f0; }
          50% { text-shadow: 2px -2px 10px #0f0, -2px 0 20px #0f0; }
          75% { text-shadow: -2px 2px 10px #0f0, 0 -2px 20px #0f0; }
        }
        
        .glitch {
          animation: glitch 2s infinite;
        }
      `}</style>
    </div>
  );
};

export default VoiceAIAssistant;