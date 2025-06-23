
import React from 'react';

interface CodeEditorProps {
  value: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  readOnly?: boolean;
  label: string;
  id: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  onChange,
  placeholder,
  readOnly = false,
  label,
  id
}) => {
  return (
    <div className="flex flex-col h-full">
      <label htmlFor={id} className="block text-sm font-medium text-slate-300 mb-1">
        {label}
      </label>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange && onChange(e.target.value)}
        placeholder={placeholder}
        readOnly={readOnly}
        className={`w-full flex-grow p-3 font-mono text-sm 
                   bg-slate-800/60 border border-slate-700/80 rounded-md shadow-inner shadow-slate-950/30 
                   focus-within:border-purple-500/80 focus-within:ring-1 focus-within:ring-purple-500/80 focus-within:shadow-md
                   resize-none transition-all duration-150
                   ${readOnly ? 'text-slate-400 cursor-not-allowed' : 'text-slate-100 placeholder-slate-500'}`}
        spellCheck="false"
      />
    </div>
  );
};

export default CodeEditor;