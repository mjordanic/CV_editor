"""Configuration loader for agent models.

This module provides utilities to load and access agent configuration
from the YAML configuration file.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Cache for loaded configuration to avoid repeated file reads
_config_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> str:
    """
    Get the path to the agent_models.yaml configuration file.
    
    Returns:
        str: Absolute path to the configuration file
    """
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "agent_models.yaml")
    return config_path


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load the agent configuration from YAML file.
    
    Args:
        force_reload: If True, reload from file even if cached
        
    Returns:
        dict: Configuration dictionary with agent settings
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
    """
    global _config_cache
    
    # Return cached config if available and not forcing reload
    if _config_cache is not None and not force_reload:
        logger.debug("Returning cached configuration")
        return _config_cache
    
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Agent configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config or 'agents' not in config:
            logger.error("Invalid configuration file: missing 'agents' key")
            raise ValueError("Invalid configuration file: missing 'agents' key")
        
        _config_cache = config
        logger.info(f"Configuration loaded successfully from {config_path}")
        logger.debug(f"Loaded configuration for {len(config['agents'])} agents")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def get_agent_config(agent_name: str, default_model: Optional[str] = None, 
                     default_temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., 'cv_writer', 'router')
        default_model: Fallback model if config unavailable or agent not found
        default_temperature: Fallback temperature if config unavailable or agent not found
        
    Returns:
        dict: Agent configuration with 'model' and 'temperature' keys
        
    Example:
        >>> config = get_agent_config('cv_writer', 'openai:gpt-4o-mini', 0.2)
        >>> print(config['model'])
        'openai:gpt-4o-mini'
    """
    try:
        config = load_config()
        agents = config.get('agents', {})
        
        if agent_name not in agents:
            logger.warning(f"Agent '{agent_name}' not found in configuration, using defaults")
            return {
                'model': default_model,
                'temperature': default_temperature
            }
        
        agent_config = agents[agent_name].copy()  # Make a copy to avoid modifying cached config
        
        # Apply defaults for model and temperature if not specified in config. An explicit
        # `temperature: null` means "the model's default" (some models accept no other value).
        if agent_config.get('model') is None:
            agent_config['model'] = default_model
        if 'temperature' not in agent_config:
            agent_config['temperature'] = default_temperature
        
        _apply_model_override(agent_name, agent_config)
        logger.debug(f"Loaded config for '{agent_name}': model={agent_config.get('model')}, temperature={agent_config.get('temperature')}")
        
        # Return complete config with all fields from YAML
        return agent_config
        
    except Exception as e:
        logger.warning(f"Error loading configuration for '{agent_name}': {e}. Using defaults.")
        return {
            'model': default_model,
            'temperature': default_temperature
        }


MODEL_OVERRIDE_ENV = "CV_EDITOR_MODEL"
GROUNDING_MODEL_OVERRIDE_ENV = "CV_EDITOR_GROUNDING_MODEL"


def _apply_model_override(agent_name: str, agent_config: Dict[str, Any]) -> None:
    """
    If CV_EDITOR_MODEL is set (e.g. "ollama:gemma4:e4b" or "openai:gpt-6-luna"), use that model for
    every agent that uses one, in place of the models in agent_models.yaml. Handy for local runs
    and for comparing models without editing the file.

    The grounding check is exempt: it is meant to run on a different model from the writer's, so
    only CV_EDITOR_GROUNDING_MODEL replaces its model.
    """
    env = GROUNDING_MODEL_OVERRIDE_ENV if agent_name == "grounding" else MODEL_OVERRIDE_ENV
    override = os.environ.get(env, "").strip()
    if not override:
        return
    for key in ("model", "filter_model"):
        if agent_config.get(key):
            agent_config[key] = override
    # Some models accept only their default temperature: CV_EDITOR_TEMPERATURE=default leaves it unset.
    temperature = os.environ.get("CV_EDITOR_TEMPERATURE", "").strip()
    if temperature and agent_config.get("model"):
        for key in ("temperature", "filter_temperature"):
            if key in agent_config:
                agent_config[key] = None if temperature == "default" else float(temperature)


def local_model_kwargs(model: Optional[str]) -> Dict[str, Any]:
    """
    Extra init_chat_model arguments for a local Ollama model ("ollama:<name>"), empty otherwise.

    Ollama serves models with a 4096-token context by default, which cuts off this tool's prompts
    (CV + job ad + portfolio), so ask for a larger one (CV_EDITOR_OLLAMA_NUM_CTX, default 16384).
    Thinking is turned off: on a laptop it multiplies the time per call.
    """
    if not model or not model.startswith("ollama:"):
        return {}
    return {"num_ctx": int(os.environ.get("CV_EDITOR_OLLAMA_NUM_CTX", "16384")), "reasoning": False}


def get_all_agents() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration for all agents.
    
    Returns:
        dict: Dictionary mapping agent names to their configurations
    """
    try:
        config = load_config()
        return config.get('agents', {})
    except Exception as e:
        logger.error(f"Error loading all agent configurations: {e}")
        return {}
