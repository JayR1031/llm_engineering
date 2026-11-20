from typing import Optional, List, Dict
from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from openai import OpenAI
import json


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-6.2-sol"
    REASONING_EFFORT = "medium"

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool to scan
        """
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        """
        Run the tool to estimate true value
        """
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> Dict:
        """
        Run the tool to notify the user
        """
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
        else:
            self.log("Autonomous Planning agent is notifying user")
            self.messenger.notify(description, deal_price, estimated_true_value, url)
            deal = Deal(product_description=description, price=deal_price, url=url)
            discount = estimated_true_value - deal_price
            self.opportunity = Opportunity(
                deal=deal, estimate=estimated_true_value, discount=discount
            )
        return "Notification sent ok"

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal scraped from the internet",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value that this is worth",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this deal as scraped from the internet",
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the json for the tools to be used with the OpenAI Responses API
        """
        return [
            {"type": "function", **self.scan_function},
            {"type": "function", **self.estimate_function},
            {"type": "function", **self.notify_function},
        ]

    def handle_tool_call(self, tool_calls):
        """
        Actually call the tools associated with these Responses API function_call items
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = json.loads(tool_call.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result if isinstance(result, str) else json.dumps(result),
                }
            )
        return results

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        input_items = self.messages[:]
        done = False
        while not done:
            response = self.openai.responses.create(
                model=self.MODEL,
                reasoning={"effort": self.REASONING_EFFORT},
                input=input_items,
                tools=self.get_tools(),
            )
            input_items.extend(response.output)
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if tool_calls:
                results = self.handle_tool_call(tool_calls)
                input_items.extend(results)
            else:
                done = True
        reply = response.output_text
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
