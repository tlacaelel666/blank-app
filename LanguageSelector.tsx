
import React from 'react';
import { Language, LanguageOption } from '../types';
import { SUPPORTED_LANGUAGES } from '../constants';

interface LanguageSelectorProps {
  id: string;
  value: Language;
  onChange: (value: Language) => void;
  label: string;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({ id, value, onChange, label }) => {
  return (
    <div className="flex-1">
      <label htmlFor={id} className="block text-sm font-medium text-slate-300 mb-1">
        {label}
      </label>
      <select
        id={id}
        name={id}
        value={value}
        onChange={(e) => onChange(e.target.value as Language)}
        className="mt-1 block w-full pl-3 pr-10 py-2 text-base 
                   bg-slate-700/80 border border-slate-600 hover:border-purple-500/70 focus:border-purple-500 
                   text-slate-200 placeholder-slate-500
                   focus:outline-none focus:ring-1 focus:ring-purple-500 
                   sm:text-sm rounded-md shadow-sm transition-colors duration-150"
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang.value} value={lang.value} className="bg-slate-700 text-slate-200">
            {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LanguageSelector;