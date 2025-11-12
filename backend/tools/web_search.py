from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Optional


def get_search_web_ddg() -> DuckDuckGoSearchRun:
    search = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(
            max_results=5,
            region="us-en",
        )
    )
    return search
