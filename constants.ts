
import { Language, LanguageOption } from './types';

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { value: Language.JAVASCRIPT, label: "JavaScript" },
  { value: Language.PYTHON, label: "Python" },
  { value: Language.JAVA, label: "Java" },
  { value: Language.TYPESCRIPT, label: "TypeScript" },
  { value: Language.GO, label: "Go" },
  { value: Language.RUBY, label: "Ruby" },
  { value: Language.CSHARP, label: "C#" },
  { value: Language.CPP, label: "C++" },
  { value: Language.KOTLIN, label: "Kotlin" },
  { value: Language.SWIFT, label: "Swift" },
  { value: Language.PHP, label: "PHP" },
  { value: Language.RUST, label: "Rust" },
];

export const GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17";
