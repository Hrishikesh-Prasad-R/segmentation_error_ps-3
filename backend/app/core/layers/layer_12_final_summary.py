"""
LAYER 12: FINAL SUMMARY LAYER (GenAI)
Purpose: Generate a comprehensive executive summary AFTER all analysis is complete.
Type: AI-Assisted

Features:
- Accesses full pipeline context (decisions, responsibility, logs)
- Uses Google Gemini for natural language generation
- Falls back to deterministic templates if API unavailable
- Provides actionable insights based on the complete picture
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Google Gemini API (New SDK)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


@dataclass
class SummaryContext:
    """Context for generating the final summary."""
    dqs: float
    overall_status: str
    decision_reasoning: Dict[str, Any]
    dimensions: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    confidence_band: str
    confidence_score: int
    responsible_party: str
    next_steps: List[str]
    liability_summary: str
    layer_trace: List[Dict[str, Any]]
    logs: List[str]


class Layer12FinalSummary:
    """
    LAYER 12: FINAL SUMMARY LAYER
    Purpose: Generate executive summary with full pipeline visibility.
    Type: AI-Assisted with deterministic fallback.
    
    This layer runs AFTER all other layers, giving it access to:
    - Complete decision and reasoning
    - Responsibility assignment
    - Full audit trail and logs
    """
    
    def __init__(self):
        self._genai_enabled = True
        
    def summarize(self, context: SummaryContext) -> tuple[str, Dict[str, Any]]:
        """
        Generate final executive summary using all available context.
        Returns: (summary_text, metadata)
        """
        metadata = {
            "genai_enabled": self._genai_enabled,
            "template_fallback": False,
            "gemini_available": GEMINI_AVAILABLE
        }
        
        # Try to use Google Gemini API (New SDK)
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        
        if GEMINI_AVAILABLE and api_key:
            try:
                client = genai.Client(api_key=api_key)
                
                # Build comprehensive prompt with full context
                prompt = self._build_comprehensive_prompt(context)
                
                # Call Gemini API (new SDK pattern)
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                summary = response.text
                
                metadata["model"] = "gemini-2.0-flash"
                metadata["api_used"] = True
                
                return summary, metadata
                
            except Exception as e:
                # FIREBREAK: Fall back to template on API error
                metadata["template_fallback"] = True
                metadata["error"] = str(e)
                metadata["reason"] = "API call failed"
        else:
            metadata["template_fallback"] = True
            if not GEMINI_AVAILABLE:
                metadata["reason"] = "google-generativeai not installed"
            elif not api_key:
                metadata["reason"] = "GEMINI_API_KEY not set"
        
        # Fallback to enhanced template
        summary = self._generate_template_summary(context)
        metadata["model"] = "template_v4_comprehensive"
        
        return summary, metadata
    
    def _build_comprehensive_prompt(self, context: SummaryContext) -> str:
        """Build a detailed prompt for Gemini API with full context."""
        # Format dimension scores
        dim_text = "\n".join([
            f"  - {d.get('dimension', 'Unknown')}: {d.get('score', 0):.1f}% ({d.get('status', 'N/A')})"
            for d in context.dimensions
        ])
        
        # Format anomaly summary
        high_sev = sum(1 for a in context.anomalies if a.get('severity') in ['HIGH', 'CRITICAL'])
        med_sev = sum(1 for a in context.anomalies if a.get('severity') == 'MEDIUM')
        low_sev = sum(1 for a in context.anomalies if a.get('severity') == 'LOW')
        
        # Format layer trace
        trace_text = "\n".join([
            f"  - {t.get('layer', 'Unknown')}: {t.get('status', 'N/A')} - {t.get('details', '')}"
            for t in context.layer_trace
        ])
        
        # Format decision reasoning
        reasoning = context.decision_reasoning
        rule_triggers = reasoning.get('rule_triggers', [])
        ai_inputs = reasoning.get('ai_inputs', [])
        
        prompt = f"""You are a Data Quality Expert AI. Analyze this complete pipeline execution report and provide an executive summary.

## COMPLETE PIPELINE EXECUTION REPORT

### Overall Results
**Composite DQS Score**: {context.dqs:.1f}/100
**Final Decision**: {context.overall_status}
**Confidence**: {context.confidence_band} ({context.confidence_score}%)

### Dimension Breakdown
{dim_text}

### Anomaly Detection
- Total Anomalies: {len(context.anomalies)}
- High Severity: {high_sev}
- Medium Severity: {med_sev}
- Low Severity: {low_sev}

### Decision Reasoning
**Primary Reason**: {reasoning.get('primary_reason', 'N/A')}
**Rule Triggers**: {', '.join(rule_triggers) if rule_triggers else 'None'}
**AI Inputs**: {', '.join(ai_inputs) if ai_inputs else 'None'}

### Responsibility Assignment
**Responsible Party**: {context.responsible_party}
**Liability**: {context.liability_summary}

### Next Steps
{chr(10).join(['- ' + step for step in context.next_steps])}

### Pipeline Execution Trace
{trace_text}

## INSTRUCTIONS
Provide a concise 4-5 sentence executive summary that:
1. States the overall data quality assessment (Excellent/Acceptable/Concerning/Critical)
2. Explains WHY this decision was made (cite the primary rule or AI input)
3. Highlights key metrics (worst dimension, anomaly counts)
4. States who is responsible and what action is required
5. Gives a clear, actionable recommendation

Keep the response professional and actionable. Do not use markdown formatting."""
        
        return prompt
    
    def _generate_template_summary(self, context: SummaryContext) -> str:
        """Generate comprehensive template summary when Gemini is unavailable."""
        parts = []
        
        # Overall assessment
        dqs = context.dqs
        if dqs >= 90:
            parts.append(f"Excellent Data Quality (DQS: {dqs:.1f}/100).")
        elif dqs >= 75:
            parts.append(f"Good Data Quality (DQS: {dqs:.1f}/100).")
        elif dqs >= 60:
            parts.append(f"Acceptable Data Quality (DQS: {dqs:.1f}/100).")
        elif dqs >= 40:
            parts.append(f"Poor Data Quality (DQS: {dqs:.1f}/100).")
        else:
            parts.append(f"Critical Data Quality Issues (DQS: {dqs:.1f}/100).")
        
        # Decision explanation
        reasoning = context.decision_reasoning
        parts.append(f"Decision: {context.overall_status}. {reasoning.get('primary_reason', '')}")
        
        # Find worst dimension
        if context.dimensions:
            worst = min(context.dimensions, key=lambda d: d.get('score', 100))
            if worst.get('score', 100) < 80:
                parts.append(f"Lowest dimension: {worst.get('dimension', 'Unknown')} at {worst.get('score', 0):.1f}%.")
        
        # Anomaly summary
        if context.anomalies:
            high_sev = sum(1 for a in context.anomalies if a.get('severity') in ['HIGH', 'CRITICAL'])
            if high_sev > 0:
                parts.append(f"{high_sev} high-severity anomalies require investigation.")
            else:
                parts.append(f"{len(context.anomalies)} minor anomalies flagged.")
        
        # Responsibility
        parts.append(f"Responsible: {context.responsible_party}. {context.liability_summary}")
        
        # Next steps
        if context.next_steps:
            parts.append(f"Next: {context.next_steps[0]}")
        
        return " ".join(parts)
