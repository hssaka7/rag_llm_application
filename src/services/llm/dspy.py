
import dspy

lm = dspy.Ollama(model="gemma3:1b")
dspy.configure(lm=lm)