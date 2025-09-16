import asyncio
import json
import logging
from typing import Callable, Awaitable
import httpx
from pydantic import BaseModel, Field
from websocket_manager import WebSocketManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AuthResponse(BaseModel):
    token: str
    user_id: str = Field(..., alias="userId")
    resource: str


class IncomingResponse(BaseModel):
    engagement_id: str = Field(..., alias="engagementId")
    zm_aid: str = Field(..., alias="zmAid")


class EngagementManager:
    """
    Manages the entire lifecycle of a single customer support engagement.
    """

    def __init__(
        self,
        base_api_url: str,
        flow_id: str,
        on_agent_message_callback: Callable[[dict | str], Awaitable[None]],
    ):
        """
        Initializes the EngagementManager for a new conversation.

        Args:
            base_api_url (str): The base URL for the Zoom Contact Center APIs.
            flow_id (str): The specific flow ID for initiating the engagement.
            on_agent_message_callback (Callable): An async callback function to send
                                                   messages/events back to the primary orchestrator.
        """
        self.base_api_url = base_api_url
        self.zoom_api_key = os.environ.get("ZOOM_API_KEY")
        self.flow_id = flow_id
        self.on_agent_message_callback = on_agent_message_callback

        self.http_client = httpx.AsyncClient()
        self.ws_manager: WebSocketManager | None = None

        # State for the engagement
        self.auth_token: str | None = None
        self.user_id: str | None = None
        self.user_email: str | None = None
        self.user_name: str | None = None
        self.resource: str | None = None
        self.engagement_id: str | None = None
        self.chat_session_id: str | None = None
        self.zm_aid: str | None = None
        self.is_agent_connected = False

    async def _handle_websocket_message(self, message: str):
        """Internal callback to process messages from the WebSocketManager."""
        # Ignore application-level ping/pong messages
        if message.startswith("ping") or message.startswith("pong"):
            logger.info(f"Received heartbeat from server: {message}")
            return

        if "agents are busy" in message.lower():
            logger.warning("All agents are busy. Notifying orchestrator.")
            await self.on_agent_message_callback(
                {"event": "no_agents_available", "details": message}
            )
            return

        try:
            data = json.loads(message)
            logger.info(f"Received structured message from WebSocket: {data}")

            event = data.get("event")
            if (
                event == "agent_connected"
            ):  # Need to update according to the actual event name
                logger.info("Agent has connected to the engagement.")
                self.is_agent_connected = True
                asyncio.create_task(self.send_connection_ack())
                # Also forward this event to the main orchestrator
                await self.on_agent_message_callback(data)
            elif event == "agent_message" or event == "typing":
                await self.on_agent_message_callback(data)
            else:
                logger.debug(f"Received unhandled event type: {event}")
                await self.on_agent_message_callback(data)  # Forward unknown events too

        except json.JSONDecodeError:
            logger.warning(
                f"Received a non-JSON message that was not handled: {message}"
            )

    async def get_auth_token(self) -> bool:
        """Calls the auth endpoint to retrieve a JWT token for the session."""
        url = f"{self.base_api_url}/v1/auth/token/generate/in/visitor/mode"
        payload = {
            "customerId": null,
            "customerName": self.user_name or "Guest User",
            "email": self.user_email,
            "apiKey": self.zoom_api_key,
        }
        try:
            logger.info("Requesting auth token...")
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()

            auth_data = AuthResponse(**response.json())
            self.auth_token = auth_data.result.token
            self.user_id = auth_data.result.customer.id
            self.resource = auth_data.resource
            self.zm_aid = auth_data.result.zmAid
            logger.info("Successfully retrieved auth token.")
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error getting auth token: {e}")
            return False

    async def initiate_engagement(self) -> bool:
        """
        Orchestrates the full engagement initiation process:
        1. Gets an auth token.
        2. Calls the 'incoming' API to get engagement details.
        3. Establishes the WebSocket connection.
        """
        if not await self.get_auth_token():
            return False

        url = f"{self.base_api_url}/v1/livechat/customer/incoming"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {
            entryPointType: "entryId",
            entryPoint: self.zoom_api_key,
            preChatSurveyCustomerInfo: {
                nickName: self.user_name or "Guest User",
                email: self.user_email,
            },
            featureOption: "7",
            customerBrowserInfo: {
                deviceDetailInDTO: {
                    "deviceId": "",
                    "deviceName": "",
                    "deviceType": "",
                    "deviceModel": "",
                    "os": "Windows 10",
                    "browser": "Chrome",
                    "browserVersion": "139",
                    "mobileAppName": "",
                    "source": "Web",
                },
                "referrer": "",
                "engagementStartPage": "",
                "engagementStartPageTitle": "",
            },
            consumerWebsiteData: [],
        }
        try:
            logger.info("Initiating engagement...")
            response = await self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            incoming_data = IncomingResponse(**response.json())
            self.engagement_id = incoming_data.result.engagementId
            logger.info(f"Engagement created with ID: {self.engagement_id}")

            # Now, establish the WebSocket connection
            self.ws_manager = WebSocketManager(
                zm_aid=self.zm_aid, on_message_callback=self._handle_websocket_message
            )
            await self.ws_manager.connect(self.auth_token, self.user_id, self.resource)
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error initiating engagement: {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during WebSocket connection: {e}"
            )
            return False

    async def send_connection_ack(self):
        """Sends the 'connected' acknowledgement after an agent joins."""
        if not self.engagement_id or not self.auth_token:
            logger.error(
                "Cannot send connection ack: missing engagement_id or auth_token."
            )
            return

        url = f"{self.base_api_url}/v1/livechat/customer/connected"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {"engagementId": self.engagement_id}
        try:
            logger.info("Sending connection acknowledgement...")
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            self.chat_session_id = response.json().get("result", {}).get("chatSessionId")
            if not self.chat_session_id:
                logger.error("No chatSessionId returned in connection ack response.")
            return response.status
            logger.info("Connection acknowledgement sent successfully.")

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error sending connection acknowledgement: {e}")

    async def send_message(self, text: str):
        """Sends a message from the customer to the Zoom agent."""
        if not self.engagement_id or not self.auth_token:
            logger.error("Cannot send message: missing engagement_id or auth_token.")
            return

        if not self.is_agent_connected:
            logger.warning(
                "Attempted to send message before agent was connected. Message not sent."
            )
            return

        url = f"{self.base_api_url}/v1/livechat/message/send"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        payload = {"engagementId": self.engagement_id, "message": {"text": text}}
        try:
            logger.info(f"Sending message to agent: '{text}'")
            response = await self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info("Message sent successfully.")
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error sending message: {e}")

    async def formatAndSendChatHistory(self, chat_history: list[dict]):
        formatted_chat_history = []
        for message in chat_history:
            sender = message["sender"]
            text = message["text"]
            formatted_text = f"[{sender}]: {text}"
            formatted_chat_history.append({"text": formatted_text})

        for message in formatted_chat_history:
            await self.send_message(message["text"])

    async def close(self):
        """Gracefully closes all connections."""
        logger.info("Closing engagement manager...")
        if self.ws_manager:
            await self.ws_manager.close()
        await self.http_client.aclose()
        logger.info("Engagement manager closed.")