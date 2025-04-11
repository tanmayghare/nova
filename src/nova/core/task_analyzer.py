from enum import Enum
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AutomationType(Enum):
    HTML = "html"
    VISUAL = "visual"
    HYBRID = "hybrid"

class TaskAnalyzer:
    """Analyzes tasks to determine the best automation approach."""
    
    # Keywords that suggest visual analysis might be needed
    VISUAL_KEYWORDS = {
        'layout', 'appearance', 'color', 'style', 'position', 'visible',
        'hidden', 'display', 'render', 'screenshot', 'image', 'compare',
        'visual', 'look', 'see', 'check if', 'verify appearance', 'align',
        'design', 'mockup', 'banner', 'hero', 'logo'
    }
    
    # Keywords that suggest HTML parsing would be better
    HTML_KEYWORDS = {
        'text', 'content', 'value', 'input', 'select', 'click', 'type',
        'submit', 'form', 'table', 'list', 'menu', 'link', 'button',
        'extract', 'get', 'find', 'data', 'information', 'price',
        'product', 'catalog', 'search', 'filter', 'sort'
    }
    
    @classmethod
    def analyze_task(cls, task: str) -> Dict[str, Any]:
        """Analyze a task and determine the best automation approach.
        
        Args:
            task: The task description to analyze
            
        Returns:
            Dictionary containing analysis results including:
            - approach: The recommended automation type
            - visual_score: Score for visual analysis suitability
            - html_score: Score for HTML parsing suitability
            - requires_visual: Whether visual analysis is required
            - requires_html: Whether HTML parsing is required
        """
        task_lower = task.lower()
        
        # Count keyword matches
        visual_score = sum(1 for keyword in cls.VISUAL_KEYWORDS if keyword in task_lower)
        html_score = sum(1 for keyword in cls.HTML_KEYWORDS if keyword in task_lower)
        
        # Determine specific requirements
        requires_visual = any(keyword in task_lower for keyword in [
            'screenshot', 'compare', 'visual regression', 'layout',
            'verify appearance', 'check alignment', 'design mockup'
        ])
        
        requires_html = any(keyword in task_lower for keyword in [
            'extract', 'table', 'form', 'text content', 'data',
            'information', 'price', 'product details'
        ])
        
        # Decision logic
        if requires_visual and requires_html:
            approach = AutomationType.HYBRID
        elif visual_score > html_score or requires_visual:
            approach = AutomationType.VISUAL
        else:
            approach = AutomationType.HTML
            
        logger.info(
            f"Task analysis: approach={approach.value}, "
            f"visual_score={visual_score}, html_score={html_score}, "
            f"requires_visual={requires_visual}, requires_html={requires_html}"
        )
            
        return {
            "approach": approach,
            "visual_score": visual_score,
            "html_score": html_score,
            "requires_visual": requires_visual,
            "requires_html": requires_html
        } 