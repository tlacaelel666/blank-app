
import React from 'react';

interface ErrorAlertProps {
  message: string;
  onDismiss?: () => void;
}

const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onDismiss }) => {
  if (!message) return null;
  return (
    <div
      className="bg-red-200/20 backdrop-blur-sm border-l-4 border-red-500 text-red-300 p-4 my-4 rounded-md shadow-lg shadow-red-900/30"
      role="alert"
    >
      <div className="flex items-start">
        <div className="py-1">
          <svg className="fill-current h-6 w-6 text-red-400 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9 9a1 1 0 011-1h0a1 1 0 011 1v4a1 1 0 01-1 1h0a1 1 0 01-1-1V9zm1-4a1 1 0 100-2 1 1 0 000 2z"/></svg>
        </div>
        <div className="flex-grow">
          <p className="font-bold text-red-400">Error</p>
          <p className="text-sm text-red-300">{message}</p>
        </div>
        {onDismiss && (
          <button onClick={onDismiss} className="ml-4 text-red-400 hover:text-red-200 transition-colors">
            <svg className="fill-current h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M10 8.586L2.929 1.515 1.515 2.929 8.586 10l-7.071 7.071 1.414 1.414L10 11.414l7.071 7.071 1.414-1.414L11.414 10l7.071-7.071-1.414-1.414L10 8.586z"/></svg>
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorAlert;