from langchain_core.tools import StructuredTool
from .invoices import create_paypal_invoice, send_paypal_invoice, CreateInvoiceSchema, SendInvoiceSchema
from .disputes import provide_dispute_evidence, ProvideEvidenceSchema, list_paypal_disputes

PAYPAL_TOOL_MAP = {
    "create_paypal_invoice": StructuredTool.from_function(
        func=create_paypal_invoice,
        name="create_paypal_invoice",
        description="Creates a draft invoice in PayPal. Use this when the user wants to bill a customer.",
        args_schema=CreateInvoiceSchema
    ),
    "send_paypal_invoice": StructuredTool.from_function(
        func=send_paypal_invoice,
        name="send_paypal_invoice",
        description="Sends an existing PayPal invoice to the customer.",
        args_schema=SendInvoiceSchema
    ),
    "provide_dispute_evidence": StructuredTool.from_function(
        func=provide_dispute_evidence,
        name="provide_dispute_evidence",
        description="Submits evidence for an open PayPal dispute.",
        args_schema=ProvideEvidenceSchema
    ),
    "list_paypal_disputes": StructuredTool.from_function(
        func=list_paypal_disputes,
        name="list_paypal_disputes",
        description="Fetch all customer disputes for the merchant account."
    )
}
