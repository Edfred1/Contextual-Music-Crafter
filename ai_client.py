import os
import logging
import time

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False

# Try importing OpenAI (for DeepSeek)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

class UnifiedResponse:
    def __init__(self, text_content):
        self._text = text_content
        
    @property
    def text(self):
        return self._text

class UnifiedModel:
    def __init__(self, provider, model_name, api_key=None, system_instruction=None, generation_config=None, use_thinking=False):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.generation_config = generation_config or {}
        self.use_thinking = use_thinking # New flag for thinking model
        
        self._gemini_model = None
        self._openai_client = None
        
        if self.provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("google-generativeai library not found. Please install it.")
            # For Gemini, we assume genai.configure() is called globally or we configure it here if key provided
            if self.api_key:
                genai.configure(api_key=self.api_key)
            
            self._gemini_model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                system_instruction=self.system_instruction
            )
            
        elif self.provider == "deepseek":
            if not HAS_OPENAI:
                raise ImportError("openai library not found. Please install it with 'pip install openai'.")
            
            # DeepSeek uses OpenAI client compatible API
            # Base URL for DeepSeek is usually https://api.deepseek.com
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

    def generate_content(self, prompt, generation_config=None, safety_settings=None):
        # Allow overriding config per call
        config = generation_config or self.generation_config
        
        if self.provider == "gemini":
            # Just delegate
            return self._gemini_model.generate_content(prompt, generation_config=config, safety_settings=safety_settings)
            
        elif self.provider == "deepseek":
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
                
            # Handle prompt (can be string or list of parts in Gemini)
            final_prompt = prompt
            if isinstance(prompt, list):
                # Convert list parts to string for DeepSeek (assuming text parts)
                # If there are images/blobs, this simple conversion won't work, but DeepSeek is mostly text?
                # Actually DeepSeek VL exists but standard DeepSeek-V3 is text.
                # For now assume text-only or simple conversion.
                final_prompt = "\n".join([str(p) for p in prompt])
            
            messages.append({"role": "user", "content": final_prompt})
            
            # Map parameters
            # Gemini: max_output_tokens, temperature, top_p, top_k, stop_sequences
            # OpenAI: max_tokens, temperature, top_p, stop
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            
            if config:
                if "max_output_tokens" in config:
                    # DeepSeek max_tokens hard limit is 8192 for the output
                    max_tokens = config["max_output_tokens"]
                    if max_tokens > 8192:
                        max_tokens = 8192
                    kwargs["max_tokens"] = max_tokens
                if "temperature" in config:
                    kwargs["temperature"] = config["temperature"]
                if "top_p" in config:
                    kwargs["top_p"] = config["top_p"]
                if "stop_sequences" in config:
                    kwargs["stop"] = config["stop_sequences"]
                # Handle structured JSON output mapping
                # Note: deepseek-reasoner does not support response_format='json_object' well, it prefers raw text.
                if config.get("response_mime_type") == "application/json":
                    if "reasoner" not in self.model_name:
                        kwargs["response_format"] = {"type": "json_object"}
            
            try:
                response = self._openai_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return UnifiedResponse(content)
            except Exception as e:
                # Wrap or re-raise
                logging.error(f"DeepSeek generation failed: {e}")
                raise e

def create_model(config, model_name_override=None, use_thinking=False):
    """Factory to create the appropriate model based on config."""
    # This helper reads from the standard config dict structure used in this project
    
    # Determine provider
    # Config might have specific provider field or we infer from model name or existing fields
    provider = config.get("provider", "gemini").lower() # Default to gemini, normalize to lowercase
    
    # Determine model name
    model_name = model_name_override or config.get("model_name", "gemini-2.5-flash")
    if provider == "deepseek":
        # Always use deepseek-chat (reasoner has issues with meaningful output)
        if "gemini" in model_name:
            # Fallback if user switched provider but didn't change model name
            model_name = "deepseek-chat"
        
    # API Key
    # The project handles rotation for Gemini. 
    # For DeepSeek, we look for 'deepseek_api_key' or use the 'api_key' if user put it there (less safe if mixed).
    
    api_key = None
    if provider == "deepseek":
        api_key = config.get("deepseek_api_key")
        if not api_key:
            # Fallback to standard api_key if it looks like a deepseek key? 
            # DeepSeek keys usually start with 'sk-'. Gemini keys usually different.
            # But let's stick to explicit config first.
            pass
            
    elif provider == "gemini":
        # The main code handles Gemini key configuration globally via genai.configure() usually,
        # but we can pass it if we have the current key.
        # The calling code in song_generator.py usually manages CURRENT_KEY_INDEX.
        # We might not need to pass api_key here if genai is already configured globally.
        pass

    # Generation config
    gen_config = {
        "temperature": config.get("temperature", 0.7),
        "max_output_tokens": config.get("max_output_tokens", 8192),
    }
    
    return UnifiedModel(
        provider=provider, 
        model_name=model_name, 
        api_key=api_key, 
        generation_config=gen_config,
        use_thinking=use_thinking
    )

