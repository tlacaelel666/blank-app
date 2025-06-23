
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { GEMINI_MODEL_NAME } from '../constants';

// Directly initialize. The environment MUST provide process.env.API_KEY.
// If process.env.API_KEY is not set by the environment, an error will likely occur
// during this initialization or when an API call is made, which is the expected behavior.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const transpileCode = async (
  sourceCode: string,
  sourceLanguage: string,
  targetLanguage: string,
  objective: string
): Promise<string> => {
  // The `ai` instance is initialized above. If API_KEY was missing/invalid,
  // the GoogleGenAI constructor or a subsequent generateContent call would throw an error.

  if (!sourceCode.trim()) {
    throw new Error("Source code cannot be empty.");
  }

  const prompt = `
You are an expert code transpiler. Your primary goal is to accurately transpile code from one programming language to another, ensuring functional equivalence and adherence to the target language's idiomatic best practices.

Transpile the following ${sourceLanguage} code to ${targetLanguage}.

Context and Objective: ${objective || `Translate the code to be functionally equivalent and idiomatic in ${targetLanguage}. Ensure the transpiled code is complete and directly usable.`}

Source ${sourceLanguage} Code:
\`\`\`${sourceLanguage.toLowerCase()}
${sourceCode}
\`\`\`

Transpiled ${targetLanguage} Code:
(Important: Provide ONLY the transpiled ${targetLanguage} code below. Do NOT include any surrounding markdown like \`\`\`${targetLanguage.toLowerCase()}\`\`\`, explanations, or introductory/concluding text. The output must be purely the ${targetLanguage} code itself.)
`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: GEMINI_MODEL_NAME,
      contents: prompt,
    });
    
    const transpiledText = response.text;
    if (!transpiledText || transpiledText.trim() === "") {
        throw new Error("Received empty transpilation result from AI. The AI might not have been able to process the request or the source code was too complex.");
    }

    // Attempt to remove markdown fences if the AI includes them despite instructions.
    let cleanedCode = transpiledText.trim();
    const fenceRegex = /^```(?:[\w.-]+)?\s*([\s\S]*?)\s*```$/s;
    const match = cleanedCode.match(fenceRegex);
    if (match && match[1]) {
      cleanedCode = match[1].trim();
    }
    
    return cleanedCode;

  } catch (error) {
    console.error("Error transpiling code with Gemini:", error);
    if (error instanceof Error) {
        throw new Error(`AI transpilation failed: ${error.message}. This could be due to an issue with the API key, network, or the request itself.`);
    }
    throw new Error("An unknown error occurred during AI transpilation.");
  }
};
