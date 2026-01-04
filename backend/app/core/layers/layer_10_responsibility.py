"""
LAYER 10: RESPONSIBILITY BOUNDARY LAYER
Purpose: Define system vs human authority and liability.
Type: Policy Enforcement

Features:
- Clear accountability assignments
- Liability statements by action type
- SLA expectations
- Escalation contacts
- Audit-ready documentation
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ResponsibleParty(Enum):
    SYSTEM_AUTOMATED = "SYSTEM_AUTOMATED"
    DATA_QUALITY_ANALYST = "DATA_QUALITY_ANALYST"
    DATA_GOVERNANCE_TEAM = "DATA_GOVERNANCE_TEAM"
    DATA_ENGINEERING = "DATA_ENGINEERING"
    BUSINESS_STAKEHOLDER = "BUSINESS_STAKEHOLDER"
    EXECUTIVE_SPONSOR = "EXECUTIVE_SPONSOR"
    SECURITY_TEAM = "SECURITY_TEAM"

@dataclass
class ResponsibilityRecord:
    """Clear record of who is responsible for what."""
    action: str
    primary_responsible: ResponsibleParty
    secondary_responsible: ResponsibleParty
    system_role: str
    human_role: str
    liability_statement: str
    sla_hours: int
    escalation_path: List[ResponsibleParty]

class Layer10Responsibility:
    """
    LAYER 10: RESPONSIBILITY BOUNDARY LAYER
    Purpose: Define where system authority ends and human responsibility begins.
    Type: Policy Enforcement
    
    Key Principle: System is a DECISION SUPPORT tool.
    Final accountability always rests with human approvers.
    """
    
    # Responsibility matrix by action type
    RESPONSIBILITY_MATRIX: Dict[str, ResponsibilityRecord] = {
        "SAFE_TO_USE": ResponsibilityRecord(
            action="SAFE_TO_USE",
            primary_responsible=ResponsibleParty.SYSTEM_AUTOMATED,
            secondary_responsible=ResponsibleParty.DATA_QUALITY_ANALYST,
            system_role="Validated all quality rules, ML models, and business logic. Approved for automated processing.",
            human_role="Monitor system logs. Review periodic quality reports. Intervene if post-processing issues arise.",
            liability_statement="System has validated data quality. Liability for processing decisions remains with the approving business unit.",
            sla_hours=0,  # Immediate
            escalation_path=[ResponsibleParty.DATA_QUALITY_ANALYST]
        ),
        "REVIEW_REQUIRED": ResponsibilityRecord(
            action="REVIEW_REQUIRED",
            primary_responsible=ResponsibleParty.DATA_QUALITY_ANALYST,
            secondary_responsible=ResponsibleParty.DATA_GOVERNANCE_TEAM,
            system_role="Flagged concerns and provided detailed analysis. Decision authority transferred to human reviewer.",
            human_role="Review flagged items. Make approve/reject decision. Document rationale.",
            liability_statement="Human reviewer assumes responsibility upon reviewing flagged items. System provided best-effort analysis.",
            sla_hours=4,
            escalation_path=[ResponsibleParty.DATA_QUALITY_ANALYST, ResponsibleParty.DATA_GOVERNANCE_TEAM]
        ),
        "ESCALATE": ResponsibilityRecord(
            action="ESCALATE",
            primary_responsible=ResponsibleParty.DATA_GOVERNANCE_TEAM,
            secondary_responsible=ResponsibleParty.EXECUTIVE_SPONSOR,
            system_role="Detected critical issues beyond automated resolution. Immediate human intervention required.",
            human_role="Investigate root cause. Coordinate remediation. Make go/no-go decision.",
            liability_statement="Escalations require senior approval. All parties in escalation chain share accountability for resolution.",
            sla_hours=2,
            escalation_path=[ResponsibleParty.DATA_GOVERNANCE_TEAM, ResponsibleParty.DATA_ENGINEERING, ResponsibleParty.EXECUTIVE_SPONSOR]
        ),
        "NO_ACTION": ResponsibilityRecord(
            action="NO_ACTION",
            primary_responsible=ResponsibleParty.DATA_ENGINEERING,
            secondary_responsible=ResponsibleParty.DATA_GOVERNANCE_TEAM,
            system_role="Could not complete analysis. Data or system issues prevented assessment.",
            human_role="Fix underlying issues. Retry analysis. Do not proceed without valid assessment.",
            liability_statement="No liability can be assigned until analysis completes. Processing without analysis is prohibited.",
            sla_hours=8,
            escalation_path=[ResponsibleParty.DATA_ENGINEERING, ResponsibleParty.DATA_GOVERNANCE_TEAM]
        )
    }
    
    # Standard disclaimers
    DISCLAIMERS = {
        "general": "This system is a Decision Support Tool. Automated decisions are based on configured rules and ML models. Final authority and liability rests with human operators.",
        "ml_limitation": "ML-based anomaly detection may not catch all data quality issues. Absence of ML flags does not guarantee data correctness.",
        "audit": "All decisions and their rationale are logged for audit purposes. Logs are immutable and retained per data governance policy.",
        "scope": "This assessment covers data quality dimensions only. Business validity, regulatory compliance, and downstream impact are outside system scope.",
        "freshness": "Analysis is valid as of the timestamp shown. Data quality may change if source data is modified."
    }
    
    def __init__(self):
        self.handoff_timestamp: str = ""
        
    def get_handoff(self, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate complete handoff documentation.
        Returns: Handoff package with accountability, SLAs, and next steps.
        """
        self.handoff_timestamp = datetime.now().isoformat()
        
        # Get responsibility record
        record = self.RESPONSIBILITY_MATRIX.get(action)
        if not record:
            record = self.RESPONSIBILITY_MATRIX["NO_ACTION"]
            
        # Build handoff package
        handoff = {
            "action": action,
            "handoff_timestamp": self.handoff_timestamp,
            
            # Accountability
            "accountability": {
                "primary_responsible": record.primary_responsible.value,
                "secondary_responsible": record.secondary_responsible.value,
                "system_role": record.system_role,
                "human_role": record.human_role
            },
            
            # Liability
            "liability": {
                "statement": record.liability_statement,
                "disclaimers": self._get_relevant_disclaimers(action)
            },
            
            # SLA
            "sla": {
                "response_required_by": self._calculate_sla_deadline(record.sla_hours),
                "hours": record.sla_hours,
                "escalation_path": [p.value for p in record.escalation_path]
            },
            
            # Communication templates
            "communication": self._get_communication_template(action, context or {}),
            
            # Audit requirements
            "audit_requirements": self._get_audit_requirements(action)
        }
        
        return handoff
    
    def _get_relevant_disclaimers(self, action: str) -> List[str]:
        """Get disclaimers relevant to this action."""
        disclaimers = [self.DISCLAIMERS["general"]]
        
        if action in ["REVIEW_REQUIRED", "ESCALATE"]:
            disclaimers.append(self.DISCLAIMERS["ml_limitation"])
            
        disclaimers.append(self.DISCLAIMERS["audit"])
        disclaimers.append(self.DISCLAIMERS["scope"])
        disclaimers.append(self.DISCLAIMERS["freshness"])
        
        return disclaimers
    
    def _calculate_sla_deadline(self, hours: int) -> str:
        """Calculate SLA deadline from now."""
        if hours == 0:
            return "IMMEDIATE"
        
        from datetime import timedelta
        deadline = datetime.now() + timedelta(hours=hours)
        return deadline.isoformat()
    
    def _get_communication_template(self, action: str, context: Dict) -> Dict[str, str]:
        """Generate communication templates for handoff."""
        templates = {
            "SAFE_TO_USE": {
                "subject": "[AUTO-APPROVED] Data Quality Assessment Passed",
                "summary": f"Dataset has been automatically approved for processing. DQS: {context.get('dqs', 'N/A')}%.",
                "action_required": "None. This is an informational notification."
            },
            "REVIEW_REQUIRED": {
                "subject": "[ACTION REQUIRED] Data Quality Review Needed",
                "summary": f"Dataset requires human review before processing. DQS: {context.get('dqs', 'N/A')}%. Anomalies: {context.get('anomalies', 'N/A')}.",
                "action_required": "Please review flagged items and approve or reject within SLA."
            },
            "ESCALATE": {
                "subject": "[URGENT] Data Quality Escalation",
                "summary": f"Critical data quality issues detected. Immediate attention required. DQS: {context.get('dqs', 'N/A')}%.",
                "action_required": "Investigate immediately. Do not proceed with processing until resolved."
            },
            "NO_ACTION": {
                "subject": "[BLOCKED] Data Quality Assessment Incomplete",
                "summary": "System could not complete data quality assessment due to data or system issues.",
                "action_required": "Fix underlying issues and retry assessment. Processing is blocked."
            }
        }
        
        return templates.get(action, templates["NO_ACTION"])
    
    def _get_audit_requirements(self, action: str) -> Dict[str, Any]:
        """Define what must be logged for audit."""
        base_requirements = {
            "required_logs": [
                "Trace ID",
                "Timestamp",
                "Input file hash",
                "All dimension scores",
                "Final decision",
                "Responsible party"
            ],
            "retention_period": "7 years",
            "tamper_protection": "Logs are append-only and cryptographically signed"
        }
        
        if action in ["REVIEW_REQUIRED", "ESCALATE"]:
            base_requirements["additional_logs"] = [
                "Reviewer identity",
                "Review timestamp",
                "Review decision",
                "Review rationale"
            ]
            
        if action == "ESCALATE":
            base_requirements["additional_logs"].extend([
                "Escalation chain",
                "Resolution timestamp",
                "Root cause analysis"
            ])
            
        return base_requirements
    
    def get_liability_summary(self, action: str) -> str:
        """Get a single-line liability summary for display."""
        summaries = {
            "SAFE_TO_USE": "✅ System approved. Processing liability with business unit.",
            "REVIEW_REQUIRED": "⚠️ Human review required. Reviewer assumes liability upon approval.",
            "ESCALATE": "🚨 Escalation required. Multi-party accountability applies.",
            "NO_ACTION": "⛔ Analysis incomplete. No processing permitted."
        }
        return summaries.get(action, summaries["NO_ACTION"])
