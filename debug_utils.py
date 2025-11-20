"""Utility module for managing debug file output."""
import os
from datetime import datetime
from typing import Optional

_debug_file_path: Optional[str] = None
_logged_message_count: int = 0  # Track how many messages have been logged


def initialize_debug_file(debug_dir: str = "debug") -> str:
    """
    Initialize the debug file with a timestamp-based filename.
    
    Args:
        debug_dir: Directory where debug file will be created
        
    Returns:
        str: Path to the debug file
    """
    global _debug_file_path
    
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create debug filename with timestamp (same format as log file)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_filename = os.path.join(debug_dir, f"cv_editor_{timestamp}.log")
    
    _debug_file_path = debug_filename
    
    # Initialize the file with a header
    global _logged_message_count
    _logged_message_count = 0  # Reset message count for new file
    
    with open(_debug_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CV EDITOR DEBUG INFORMATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    return _debug_file_path


def get_debug_file_path() -> Optional[str]:
    """
    Retrieve the active debug file path, if initialized.

    Args:
        None

    Returns:
        Optional[str]: Absolute path to the debug file or None if not initialized yet.
    """
    return _debug_file_path


def get_logged_message_count() -> int:
    """
    Return how many messages have been written to the debug log during this session.

    Args:
        None

    Returns:
        int: Count of logged messages.
    """
    return _logged_message_count


def reset_logged_message_count():
    """
    Reset the internal message counter.

    Args:
        None

    Returns:
        None
    """
    global _logged_message_count
    _logged_message_count = 0


def write_to_debug(content: str, section_title: str = ""):
    """
    Write content to the debug file.
    
    Args:
        content: Free-form text that should be appended to the debug log.
        section_title: Optional section header to prepend before the content.

    Returns:
        None
    """
    if _debug_file_path is None:
        # If not initialized, initialize with default settings
        initialize_debug_file()
    
    with open(_debug_file_path, "a", encoding="utf-8") as f:
        if section_title:
            f.write("=" * 80 + "\n")
            f.write(f"{section_title}\n")
            f.write("=" * 80 + "\n\n")
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
        f.write("\n")


def log_messages(messages: list, start_index: int = 0):
    """
    Log new messages to the debug file.
    
    Args:
        messages: List of messages to log
        start_index: Index to start logging from (for logging only new messages)
        
    Returns:
        int: Number of messages logged
    """
    global _logged_message_count
    
    if _debug_file_path is None:
        initialize_debug_file()
    
    if not messages or start_index >= len(messages):
        return 0
    
    new_messages = messages[start_index:]
    if not new_messages:
        return 0
    
    debug_content = ""
    
    for i, msg in enumerate(new_messages, start=start_index + 1):
        # Handle different message formats
        role = "unknown"
        content = ""
        
        if isinstance(msg, dict):
            # Dictionary format
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            # LangChain message object or other object format
            # Try multiple ways to get the role/type
            
            # Method 1: Check for 'type' attribute (LangChain messages)
            if hasattr(msg, "type"):
                try:
                    msg_type = msg.type
                    # Map LangChain message types to roles
                    if msg_type == "human":
                        role = "user"
                    elif msg_type == "ai":
                        role = "assistant"
                    elif msg_type == "system":
                        role = "system"
                    else:
                        role = msg_type
                except (AttributeError, TypeError):
                    pass
            
            # Method 2: Check for 'get_type' method
            if role == "unknown" and hasattr(msg, "get_type"):
                try:
                    msg_type = msg.get_type()
                    if msg_type == "human":
                        role = "user"
                    elif msg_type == "ai":
                        role = "assistant"
                    elif msg_type == "system":
                        role = "system"
                    else:
                        role = msg_type
                except (AttributeError, TypeError):
                    pass
            
            # Method 3: Try to infer from class name
            if role == "unknown" and hasattr(msg, "__class__"):
                class_name = msg.__class__.__name__.lower()
                if "human" in class_name:
                    role = "user"
                elif "ai" in class_name or "assistant" in class_name:
                    role = "assistant"
                elif "system" in class_name:
                    role = "system"
            
            # Method 4: Check if it has a 'role' attribute (some message wrappers)
            if role == "unknown" and hasattr(msg, "role"):
                try:
                    role = msg.role
                except (AttributeError, TypeError):
                    pass
            
            # Get content - try multiple methods
            if hasattr(msg, "content"):
                try:
                    content = msg.content
                except (AttributeError, TypeError):
                    content = ""
            elif hasattr(msg, "get") and callable(msg.get):
                content = msg.get("content", "")
            else:
                content = str(msg) if msg else ""
        
        debug_content += f"\n[{i}] {role.upper()}:\n"
        debug_content += "-" * 40 + "\n"
        debug_content += str(content) if content else "(empty message)"
        debug_content += "\n"
    
    debug_content += "\n"
    
    with open(_debug_file_path, "a", encoding="utf-8") as f:
        if debug_content.strip():  # Only write section header if there's content
            f.write("=" * 80 + "\n")
            f.write("MESSAGES DEBUG INFO\n")
            f.write("=" * 80 + "\n\n")
        f.write(debug_content)
    
    logged_count = len(new_messages)
    _logged_message_count += logged_count
    return logged_count

