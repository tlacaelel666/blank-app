
import React, { useState, useCallback } from 'react';
import { Language } from './types';
import { SUPPORTED_LANGUAGES } from './constants';
import LanguageSelector from './components/LanguageSelector';
import CodeEditor from './components/CodeEditor';
import LoadingIcon from './components/LoadingIcon';
import ErrorAlert from './components/ErrorAlert';
import { transpileCode } from './services/geminiService';

const App: React.FC = () => {
  const [sourceLanguage, setSourceLanguage] = useState<Language>(Language.JAVASCRIPT);
  const [targetLanguage, setTargetLanguage] = useState<Language>(Language.PYTHON);
  const [sourceCode, setSourceCode] = useState<string>('');
  const [objective, setObjective] = useState<string>('');
  const [transpiledCode, setTranspiledCode] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // State for the test panel
  const [testInput, setTestInput] = useState<string>('');
  const [testOutput, setTestOutput] = useState<string>('');
  const [isTestingCode, setIsTestingCode] = useState<boolean>(false);
  const [testError, setTestError] = useState<string | null>(null);


  const handleTranspile = useCallback(async () => {
    if (!sourceCode.trim()) {
      setError("Source code cannot be empty.");
      return;
    }
    setError(null);
    setTestOutput('');
    setTestError(null);
    setTestInput('');
    setIsLoading(true);
    setTranspiledCode('');

    try {
      const result = await transpileCode(sourceCode, sourceLanguage, targetLanguage, objective);
      setTranspiledCode(result);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred during transpilation.");
      }
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [sourceCode, sourceLanguage, targetLanguage, objective]);

  const handleSwapLanguages = () => {
    const currentSource = sourceLanguage;
    const currentTarget = targetLanguage;
    const currentSourceCode = sourceCode;
    const currentTranspiledCode = transpiledCode;

    setSourceLanguage(currentTarget);
    setTargetLanguage(currentSource);
    
    if (currentTranspiledCode.trim() !== "") {
        setSourceCode(currentTranspiledCode);
        setTranspiledCode(currentSourceCode); 
    }
    // Clear test panel when swapping
    setTestInput('');
    setTestOutput('');
    setTestError(null);
  };

  const handleRunTest = useCallback(() => {
    if (targetLanguage !== Language.JAVASCRIPT || !transpiledCode) return;

    setIsTestingCode(true);
    setTestOutput('');
    setTestError(null);
    
    const capturedLogs: string[] = [];
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;

    // Temporarily override console methods to capture logs
    console.log = (...args: any[]) => {
        capturedLogs.push(args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg, null, 2)).join(' '));
        originalConsoleLog.apply(console, args); 
    };
    console.error = (...args: any[]) => {
        capturedLogs.push("ERROR: " + args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg, null, 2)).join(' '));
        originalConsoleError.apply(console, args);
    };

    try {
        // Make testInput available as a global variable `_testInput` (as a string)
        // The user's script can then parse it if needed (e.g., JSON.parse(_testInput))
        const scriptToRun = `
            const _testInput = ${JSON.stringify(testInput)};
            ${transpiledCode}
        `;
        new Function(scriptToRun)();
        setTestOutput(capturedLogs.join('\n') || "Script executed. No output via console.log.");
    } catch (e: any) {
        setTestError(`Runtime Error: ${e.message}`);
        setTestOutput(capturedLogs.join('\n')); // Show any logs captured before the error
    } finally {
        console.log = originalConsoleLog; // Restore original console methods
        console.error = originalConsoleError;
        setIsTestingCode(false);
    }
  }, [transpiledCode, targetLanguage, testInput]);


  return (
    <div className="min-h-screen text-slate-100 p-4 sm:p-6 lg:p-8 flex flex-col items-center">
      <div 
        className="w-full max-w-7xl mx-auto bg-gradient-to-br from-slate-900 via-slate-850 to-slate-800 rounded-xl 
                   shadow-[0_20px_45px_-15px_rgba(0,0,0,0.6),_inset_0_2px_8px_rgba(168,85,247,0.15)] 
                   border border-slate-700/70 overflow-hidden"
      >
        <div className="p-4 sm:p-6 lg:p-8">
          <header className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
                AI Code Transpiler
              </h1>
              <p className="mt-2 text-lg text-slate-400">
                Translate code between languages with contextual understanding, powered by Gemini.
              </p>
            </div>
            <button
              title="My Account"
              aria-label="My Account"
              className="p-2 text-slate-400 hover:text-purple-400 transition-all duration-200 rounded-full hover:bg-slate-700/80 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-7 h-7">
                <path strokeLinecap="round" strokeLinejoin="round" d="M17.982 18.725A7.488 7.488 0 0 0 12 15.75a7.488 7.488 0 0 0-5.982 2.975m11.963 0a9 9 0 1 0-11.963 0m11.963 0A8.966 8.966 0 0 1 12 21a8.966 8.966 0 0 1-5.982-2.275M15 9.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </button>
          </header>

          {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
          
          <div className="bg-slate-800/70 backdrop-blur-sm shadow-xl shadow-slate-950/50 rounded-xl p-6 md:p-8 border border-slate-700/60">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6 items-end">
              <LanguageSelector
                id="sourceLanguage"
                label="Source Language"
                value={sourceLanguage}
                onChange={(lang) => { setSourceLanguage(lang); setTestInput(''); setTestOutput(''); setTestError(null);}}
              />
               <div className="flex items-center justify-center md:pt-7">
                  <button
                  onClick={handleSwapLanguages}
                  title="Swap Languages"
                  aria-label="Swap source and target languages"
                  className="p-2 text-slate-400 hover:text-purple-400 transition-all duration-200 rounded-full hover:bg-slate-700/80 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-slate-800"
                  >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                  </svg>
                  </button>
              </div>
              <LanguageSelector
                id="targetLanguage"
                label="Target Language"
                value={targetLanguage}
                onChange={(lang) => { setTargetLanguage(lang); setTestInput(''); setTestOutput(''); setTestError(null);}}
              />
            </div>

            <div className="mb-6">
              <label htmlFor="objective" className="block text-sm font-medium text-slate-300 mb-1">
                Code Objective/Context (Optional)
              </label>
              <input
                type="text"
                id="objective"
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                placeholder="e.g., 'Refactor for readability', 'Optimize for performance'"
                className="mt-1 block w-full px-3 py-2 bg-slate-700/80 border border-slate-600 hover:border-purple-500/70 focus:border-purple-500 rounded-md shadow-sm placeholder-slate-500 
                           focus:outline-none focus:ring-1 focus:ring-purple-500 sm:text-sm text-slate-200 transition-colors duration-150"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[300px] md:min-h-[400px] lg:min-h-[500px]">
              <CodeEditor
                id="sourceCode"
                label="Source Code"
                value={sourceCode}
                onChange={setSourceCode}
                placeholder={`Enter ${SUPPORTED_LANGUAGES.find(l => l.value === sourceLanguage)?.label || 'source'} code here...`}
              />
              <CodeEditor
                id="transpiledCode"
                label="Transpiled Code"
                value={transpiledCode}
                readOnly
                placeholder="Transpiled code will appear here..."
              />
            </div>

            <div className="mt-8 text-center">
              <button
                onClick={handleTranspile}
                disabled={isLoading || !sourceCode.trim()}
                className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold 
                           rounded-lg shadow-lg hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-slate-800 
                           transform hover:scale-105 active:scale-95 transition-all duration-150 ease-in-out 
                           disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:shadow-lg flex items-center justify-center mx-auto min-w-[180px]"
              >
                {isLoading ? (
                  <>
                    <LoadingIcon className="mr-2 w-5 h-5" />
                    Transpiling...
                  </>
                ) : (
                  "Transpile Code"
                )}
              </button>
            </div>

            {/* Test Panel for JavaScript */}
            {targetLanguage === Language.JAVASCRIPT && transpiledCode.trim() && (
              <div className="mt-8 bg-slate-800/60 backdrop-blur-sm shadow-lg shadow-slate-950/40 rounded-xl p-4 md:p-6 border border-slate-700/50">
                <h3 className="text-xl font-semibold text-slate-200 mb-4">Test Transpiled JavaScript</h3>
                
                {testError && <ErrorAlert message={testError} onDismiss={() => setTestError(null)} />}
                
                <div className="mb-4">
                  <label htmlFor="testInput" className="block text-sm font-medium text-slate-300 mb-1">
                    Test Input (available as string <code className="bg-slate-900 px-1 py-0.5 rounded text-pink-400 text-xs">_testInput</code> in your script)
                  </label>
                  <textarea
                    id="testInput"
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    placeholder={'e.g., {"name": "Test", "value": 123}\nOr simply a string/number that your script expects _testInput to be.'}
                    rows={3}
                    className="w-full p-3 font-mono text-sm bg-slate-700/80 border border-slate-600 rounded-md shadow-inner placeholder-slate-500 text-slate-100 
                               focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors duration-150"
                    spellCheck="false"
                  />
                </div>

                <button
                  onClick={handleRunTest}
                  disabled={isTestingCode}
                  className="px-6 py-2 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-md shadow focus:outline-none focus:ring-2 focus:ring-sky-500 
                             focus:ring-offset-2 focus:ring-offset-slate-800 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center min-w-[120px]"
                >
                  {isTestingCode ? (
                    <>
                      <LoadingIcon className="mr-2 w-4 h-4" /> Running...
                    </>
                  ) : (
                    "Run Script"
                  )}
                </button>

                {(testOutput || (isTestingCode && !testError)) && (
                    <div className="mt-4 flex flex-col min-h-[100px] max-h-[300px]"> {/* Container for CodeEditor to manage height */}
                        <CodeEditor
                            id="testOutput"
                            label="Output / Logs"
                            value={isTestingCode && !testOutput && !testError ? "Running script and capturing logs..." : testOutput}
                            readOnly
                            placeholder="Script output and console logs will appear here..."
                        />
                    </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        <footer className="mt-8 py-6 text-center text-sm text-slate-500 border-t border-slate-700/50">
          <p>&copy; {new Date().getFullYear()} AI Code Transpiler. Powered by Google Gemini.</p>
          <p>Results may vary. Always review and test transpiled code.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
