import asyncio
import os
import httpx
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Use FastMCP for easier tool definition with type hints and docstrings
from mcp.server.fastmcp import FastMCP
import mcp.types as types

load_dotenv()

# --- Configuration ---
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
USER_AGENT = "MCPWebUtilServer/1.0 (LanguageModelIntegration; +https://modelcontextprotocol.io)"
MAX_SEARCH_RESULTS = 5
MAX_FETCH_CHARS = 500000 # Limit fetched text size for LLM

# --- Initialize MCP Server ---
# Using FastMCP simplifies tool definition using decorators, type hints, and docstrings
mcp = FastMCP("web_search", version="1.0.0")

# --- Helper Functions ---
async def make_request(client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    """Makes an async HTTP GET request with error handling."""
    if headers is None:
        headers = {"User-Agent": USER_AGENT}
    else:
        headers["User-Agent"] = headers.get("User-Agent", USER_AGENT)

    try:
        response = await client.get(url, headers=headers, timeout=20.0, follow_redirects=True)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        return response
    except httpx.RequestError as exc:
        print(f"HTTP Request failed: {exc}")
        raise ConnectionError(f"Network error accessing {url}: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        print(f"HTTP Status error: {exc.response.status_code} for {url}")
        raise ValueError(f"HTTP error {exc.response.status_code} accessing {url}") from exc

# --- MCP Tools ---

@mcp.tool()
async def brave_search(query: str) -> str:
    """
    Performs a web search using the Brave Search API and returns formatted results.

    Args:
        query: The search query string.

    Returns:
        A formatted string containing the top search results, or an error message.
    """
    if not BRAVE_API_KEY:
        return "Error: Brave Search API key (BRAVE_API_KEY) is not configured."

    search_url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count={MAX_SEARCH_RESULTS}"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await make_request(client, search_url, headers=headers)
            data = response.json()

            if "web" not in data or "results" not in data["web"]:
                return "No web search results found."

            results = data["web"]["results"]
            if not results:
                return "No web search results found."

            formatted_results = ["Search Results:"]
            for i, result in enumerate(results[:MAX_SEARCH_RESULTS]):
                title = result.get("title", "No Title")
                url = result.get("url", "#")
                snippet = result.get("description", "No Snippet")
                formatted_results.append(f"{i+1}. {title}\n   URL: {url}\n   Snippet: {snippet}\n")

            return "\n".join(formatted_results)

        except (ConnectionError, ValueError) as e:
            return f"Error performing search: {e}"
        except Exception as e:
            print(f"Unexpected error during Brave search: {e}")
            return f"An unexpected error occurred: {e}"

@mcp.tool()
async def fetch_webpage_text(url: str) -> str:
    """
    Fetches a webpage and extracts the main text content using BeautifulSoup.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        The extracted text content (up to a limit), or an error message.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await make_request(client, url)

            # Check if content type is likely HTML
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return f"Error: Content type is not HTML ({content_type}). Cannot extract text."

            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script_or_style in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script_or_style.extract()

            # Get text and clean it up
            text = soup.get_text(separator=' ', strip=True)
            # Replace multiple spaces/newlines with a single space
            text = ' '.join(text.split())

            if not text:
                return "Could not extract any text content from the page."

            # Limit the text length
            return text[:MAX_FETCH_CHARS] + ("..." if len(text) > MAX_FETCH_CHARS else "")

        except (ConnectionError, ValueError) as e:
            return f"Error fetching webpage {url}: {e}"
        except Exception as e:
            print(f"Unexpected error fetching/parsing {url}: {e}")
            return f"An unexpected error occurred while processing {url}: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    if not BRAVE_API_KEY:
        print("Warning: BRAVE_API_KEY environment variable not set. Brave search tool will not work.")
        print("Create a .env file with BRAVE_API_KEY=YOUR_KEY")

    print("Starting Web Utils MCP Server on stdio...")
    mcp.run(transport='stdio')