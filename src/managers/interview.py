from typing import Dict, Any, List
import openai
from ..core.interfaces import InterviewManager
from ..core.config import get_settings

class OpenAIInterviewManager(InterviewManager):
    def __init__(self):
        self.settings = get_settings()
        openai.api_key = self.settings.OPENAI_API_KEY
        
        # Initialize interview state
        self.conversation_history: List[Dict[str, str]] = []
        self.question_count = 0
        self.current_topic = None
    
    async def generate_question(self, context: Dict[str, Any]) -> str:
        """Generate the next interview question based on context."""
        # Prepare the conversation context
        messages = self._prepare_conversation_context(context)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            question = response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": question
            })
            
            self.question_count += 1
            return question
            
        except Exception as e:
            print(f"Error generating question: {e}")
            return "Could you tell me more about your experience?"
    
    async def analyze_response(self,
                             response: str,
                             emotions: Dict[str, float],
                             audio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's response including text, emotions, and audio metrics."""
        # Add user response to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": response
        })
        
        # Prepare analysis prompt
        analysis_prompt = self._create_analysis_prompt(
            response, emotions, audio_metrics
        )
        
        try:
            analysis = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": analysis_prompt
                }],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse and structure the analysis
            return self._structure_analysis(analysis.choices[0].message.content)
            
        except Exception as e:
            print(f"Error analyzing response: {e}")
            return {
                "confidence": 0.5,
                "clarity": 0.5,
                "relevance": 0.5,
                "improvement_suggestions": ["Could not generate detailed analysis."]
            }