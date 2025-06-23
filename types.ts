
export enum Language {
  JAVASCRIPT = "JavaScript",
  PYTHON = "Python",
  JAVA = "Java",
  TYPESCRIPT = "TypeScript",
  GO = "Go",
  RUBY = "Ruby",
  CSHARP = "C#",
  CPP = "C++",
  KOTLIN = "Kotlin",
  SWIFT = "Swift",
  PHP = "PHP",
  RUST = "Rust",
}

export interface LanguageOption {
  value: Language;
  label: string;
}
