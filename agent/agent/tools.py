"""Tools the agent can call: order lookup and refund requests.

Refund rules (from the returns policy):
- must be within 30 days of delivery
- item must be delivered and not marked final sale
- refunds over $200 need a human to approve them
"""
import json
import re
from datetime import date
from pathlib import Path

ORDER_ID = re.compile(r"\bKG\d{4}\b", re.IGNORECASE)
RETURN_WINDOW_DAYS = 30
AUTO_APPROVE_LIMIT = 200.0


def extract_order_id(text: str) -> str | None:
    m = ORDER_ID.search(text)
    return m.group(0).upper() if m else None


class OrderTools:
    def __init__(self, orders_path: Path, today: date | None = None):
        self.orders = json.loads(orders_path.read_text(encoding="utf-8"))
        self.today = today or date.today()

    def get_order_status(self, order_id: str) -> dict:
        order = self.orders.get(order_id)
        if order is None:
            return {"ok": False, "message": f"I couldn't find order {order_id}. Please check the number."}
        status = order["status"]
        if status == "delivered":
            msg = f"Order {order_id} was delivered on {order['delivered_on']}."
        elif status == "dispatched":
            msg = (f"Order {order_id} was dispatched on {order['dispatched_on']}. "
                   f"Your tracking number is {order['tracking']}.")
        else:
            msg = f"Order {order_id} is being prepared and hasn't been dispatched yet."
        return {"ok": True, "message": msg, "order": order}

    def request_refund(self, order_id: str) -> dict:
        """Returns a decision: approved, denied or needs_human."""
        order = self.orders.get(order_id)
        if order is None:
            return {"decision": "denied", "message": f"I couldn't find order {order_id}."}
        if order["status"] != "delivered":
            return {"decision": "denied",
                    "message": (f"Order {order_id} hasn't been delivered yet, so it can't be refunded. "
                                "You can cancel it for free until it is dispatched.")}
        if order.get("final_sale"):
            return {"decision": "denied",
                    "message": f"Order {order_id} was a final sale item, so it can't be returned."}
        days = (self.today - date.fromisoformat(order["delivered_on"])).days
        if days > RETURN_WINDOW_DAYS:
            return {"decision": "denied",
                    "message": (f"Order {order_id} was delivered {days} days ago, which is outside our "
                                f"{RETURN_WINDOW_DAYS}-day return window. If the item is faulty, "
                                "you may be able to make a warranty claim.")}
        if order["total"] > AUTO_APPROVE_LIMIT:
            return {"decision": "needs_human",
                    "message": (f"Refunds over ${AUTO_APPROVE_LIMIT:.0f} are reviewed by our team. "
                                f"I've passed order {order_id} to a team member, who will be in touch "
                                "within one business day.")}
        return {"decision": "approved",
                "message": (f"Your return for order {order_id} is approved. We'll email you a prepaid "
                            "return label within one business day, and your refund of "
                            f"${order['total']:.2f} will be processed once we receive the item.")}
